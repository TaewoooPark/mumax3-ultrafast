# mumax3-for-mac — Apple Silicon 최적화: 조사 · 구현 · 실측 기록

측정 환경: **Apple M4 (10-core GPU), macOS 15.6, go 1.26.5**
기준 커밋: `b18bc5ea` → 현재 `feature/apple-silicon-metal`

이 문서는 계획서가 아니라 **실행 기록**이다. 모든 수치는 이 저장소를 빌드해 실측했고,
빗나간 가설은 지우지 않고 남겼다.

---

## 0. 가장 중요한 발견 두 가지

### (1) 병목은 커널 산술이 아니다

elementwise 커널은 대형 메시에서 **86.4 GB/s** — M4 이론치의 72%로 이미 포화다.
`float4` 벡터화(0.94×), `StorageModePrivate`(0.99×) 모두 효과 없음을 실측으로 확인했다.
**남은 개선 경로는 "패스를 빠르게"가 아니라 "패스 수를 줄이기"뿐이다.**

### (2) reduction 비결정성이 모든 최적화의 선행 조건이었다

`relax()`는 주석에 명시된 대로 **에너지가 수치 노이즈 플로어에 닿을 때까지** 반복한다.
그 에너지는 `atomicAdd` 기반 sum reduction이고, 원자적 누적 순서는 GPU 스케줄링에 의존한다.
따라서 **큐 깊이를 바꾸는 모든 최적화가 relax() 정지 시점을 옮겨 "물리 회귀"처럼 보였다.**

실측: 큐 깊이만 바뀐 두 빌드가 std4 첫 출력 샘플부터 4e-4 어긋남 — float32 노이즈 플로어
1.2e-7의 3천 배. 이걸 먼저 고치지 않으면 이후 어떤 변경도 검증할 수 없었다.

---

## 1. 착수 전 실측 베이스라인

### 1.1 step당 구조 (런타임 계측, 메시 무관)

```
kernel launches      = 95.4   ← 각각 독립 MTLComputeCommandEncoder
u32 zero-fills       = 20.2
MPSGraph FFT encodes = 36.2
command buffers      =  8.0
FULL PIPELINE DRAINS =  8.03  ← Sync 요청 16.03 + DtoH 2.00
```

### 1.2 Apple GPU 고정 비용

| 시나리오 | 비용 |
|---|---|
| 빈 command buffer commit→완료 | 8.8 µs |
| **디스패치 1개 추가** | **110.0 µs** ← GPU 웨이크업 |
| command buffer 64개 파이프라인 | 21.5 µs each |

### 1.3 디스패치 단가 (200 디스패치 × 1024 elem, min-of-60)

| 구성 | ns/dispatch |
|---|---|
| 인코더/디스패치 + CUDA식 커널 (착수 시점) | 6850 |
| 단일 인코더, serial dispatch | 6536 (**5%뿐**) |
| 단일 인코더 + native-index 커널 | 4763 |
| 단일 인코더, concurrent dispatch | 2179 |
| ICB 사전 인코딩 replay | 1577 |

→ **인코더 병합만으로는 5%.** 비용의 본질은 인코더 경계가 아니라 디스패치 간 직렬화다.

### 1.4 MPSGraph FFT 축 비대칭 (핵심)

| shape | µs | GB/s |
|---|---|---|
| {1, 512, 512} | 24.1 | 87 |
| {1, 512, 1024} | 36.6 | **115** ← 긴 축 contiguous |
| {1, 640, 640} | 60.9 | 54 |
| {1, 768, 768} | 116.7 | 40 |
| {1, 1024, 1024} | 260.4 | **32** ← 긴 축 strided, 붕괴 |

**strided 축 길이가 512를 넘으면 붕괴한다.** mesh 512²(padded 1024²)가 정확히 그 구간.

---

## 2. 구현 결과 (커밋 순서대로)

| # | 커밋 | 내용 | 물리 검증 |
|---|---|---|---|
| 1 | `0eebec69` | reduction 재현성 확보 | 동일 입력 → **편차 정확히 0**, 시간적분 bit-identical |
| 2 | `308b972d` | 정수 커널 핸들 + 미사용 pointer mask 제거 | bit-identical |
| 3 | `6f97faf3` | 복사 주변 중복 Sync 제거 | bit-identical |
| 4 | `e9cd0725` | `thread_position_in_grid` 인덱싱 | bit-identical |
| 5 | `58219144` | demag zero-padding 재초기화 제거 | bit-identical |
| 6 | `0edf1de6` | 대형 demag 변환 축별 분리 | bit-identical |
| 7 | `31924295` | max reduction SIMD 축약 + grid 확대 | bit-identical |

**7개 커밋 전부 std4 스냅샷 12개 바이트 동일, table 편차 0.000e+00.**
(1번만 예외: `relax()` 정지 시점이 한 반복 이동 — 합산 순서를 고정한 의도된 결과.
시간적분 자체는 원본 베이스라인과 바이트 동일함을 별도 확인.)

### 세부

**1. reduction 재현성** — threadgroup마다 `dst + blockIdx.x` 슬롯을 소유하게 하고 호스트가
고정 인덱스 순서로 합산. threadgroup 내부 트리는 그대로라 각 partial은 CUDA와 bit-identical,
교차 순서만 정의된다. max는 부동소수점에서 순서 무관이라 영향 없음. `Dot`은 성분별로
슬롯 블록을 분리해 launch 중첩에도 무관하게 만들었다.

**2. 정수 커널 핸들** — 디스패치마다 `C.CString` + `NSString` + `NSMutableDictionary` 해시가
돌았다. `mr_register_kernel`/`mr_launch_handle`로 1회 해석. 인자 배열은 힙 대신 고정 배열,
인코더 라벨은 `MUMAX3_METAL_LABELS` 뒤로. pointer mask는 실제로 쓰는 **13/65** 커널만 바인딩.

**3. 중복 Sync 제거** — `mr_copy`는 겹치지 않는 버퍼에 대해 이미 같은 serial queue의 async
blit이고, `mr_copy_to_host`/`_to_device`는 내부에서 드레인한다. 바깥 Sync는 전부 중복.
**드레인 8.03 → 2.07/step, command buffer 8.0 → 2.0.**

**4. native 인덱싱** — 두 형태 모두 *정확한 항등식*으로 치환:
- 3D: `blockIdx.a*blockDim.a + threadIdx.a` ≡ `thread_position_in_grid.a`
- 1D: `(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x` ≡ `gid.y*threads_per_grid.x + gid.x`
  (blockDim.y==1일 때. 원식도 threadIdx.y를 무시하므로 같은 조건에서만 옳다)

빌트인 4개 → 1개(59/65 커널). reduction 6개는 `gridDim.x*blockDim.x` 스트라이드가 필요해 유지.
`verify_launch_builtins`가 선언 없이 빌트인을 쓰는 커널을 **생성 단계에서 거부**한다.

**5. demag zero-padding** — `copyPadMul`은 문서대로 나머지를 건드리지 않으므로 패딩은 1회만
쓰면 된다. 매번 지워야 했던 이유는 역변환이 같은 버퍼를 덮었기 때문 → 역변환 전용 출력 버퍼
분리. **eval당 3 디스패치 + 3회 padded 전체 쓰기 제거 = step당 18 디스패치.**

**6. 축별 분리** — Hermitian 축과 나머지 축을 별도 변환으로. 결합형과 **차이 정확히 0**
(양방향 검증) — MPSGraph가 내부적으로 같은 분해를 하고 스케줄링만 나쁘다는 뜻.
분리는 encode가 2배라 공짜가 아니므로 **strided 축 > 512에서만** 적용.

- padded 1024²: 2467 → 2342 µs/eval (1.05×)
- padded 2048²: 11597 → 10580 µs/eval (1.10×)

게이트가 분리를 거부할 때 주 그래프가 Hermitian 축만 변환하는 버그가 있었고,
`conv_demag`의 "FFT kernel imaginary part" assertion이 잡아냈다.

**7. max reduction** — `simd_max`는 배리어 0회로 32레인 축약(트리는 512스레드에 배리어 9회).
max는 순서 무관이라 **bit-identical**. sum은 결합법칙이 없어 트리 유지 — `simd_sum`은 더
빠르지만 에너지 마지막 비트를 바꿔 relax() 정지점을 옮긴다. 매크로가 연산자 토큰을
붙여넣어 전략을 선택하므로 CUDA 소스 호출부에서 어느 쪽인지 보인다.
grid는 max만 8 → 64 (순서 무관이므로 안전, 1.3~1.6×). sum은 8 유지.

---

## 3. 최종 실측

### 3.1 end-to-end (RK45DP, 2D, best-of-4, 장시간 런)

| mesh | 착수 시점 | 현재 | 배수 |
|---|---|---|---|
| 64² | 5.802 ms | 3.956 ms | **1.47×** |
| 512² | 22.191 ms | 20.305 ms | **1.09×** |

### 3.2 demag 컨볼루션 단독 (프로세스 격리, min-of-3)

| mesh (padded) | 착수 시점 | 현재 |
|---|---|---|
| 512² (1024²) | 2467 µs/eval | 2342 µs/eval |
| 1024² (2048²) | 11597 µs/eval | 10580 µs/eval |

### 3.3 측정 신뢰도에 대한 정직한 경고

이 M4에서 **동일 바이너리의 run-to-run 편차가 최대 35%**다. 프로세스 격리, min-of-N,
장시간 런, 라운드로빈 교차를 모두 적용한 뒤에도 그렇다. 근거:
동일 코드 경로인 `nosplit`과 `gated`가 padded 256²에서 228.7 µs vs 311.5 µs로 갈렸다.

따라서:
- **~30% 미만의 개별 효과는 이 장비에서 판정 불가능하다.**
- 초기 조사에서 보고한 "Phase 0 = 1.42×"는 **노이즈였다**. 신뢰 가능한 방법론으로 재측정한 값은 1.20×다.
- 방향성이 여러 라운드에서 일관된 항목만 채택했고, FFT 임계값은 재현성 있던 대역폭
  측정에서 정했다(노이즈 큰 end-to-end 값이 아니라).

---

## 4. 검증 프로토콜

각 커밋이 통과한 것:

1. **std4 스냅샷 12개 바이트 동일 + table 절대편차** — 상대편차는 영점 교차에서 발산하므로 절대편차를 쓴다.
2. **결정성 대조** — 같은 바이너리 3회 반복. 커밋 1 이후 편차는 정확히 0.
3. **relax 없는 순수 시간적분** — `relax()`의 노이즈 플로어 정지 조건을 배제한 대조군.
   이게 커밋 1의 영향 범위를 정확히 가려냈다.
4. **해석적 반자장 계수** — 1024×1024×2 nm 박막: Nz = 0.9899, Nx = Ny = 0.0044, 합 0.9987.
   박막 이론과 일치. `conv_selftest`가 못 쓰는 상태여서 이걸 오라클로 썼다.
5. **`relax()` 수렴 케이스** — vortex 초기조건에서 평형까지, 최종 maxTorque까지 동일.
6. `go test ./cuda/...`, 생성기 self-check, 생성기 단위 테스트, 176개 `.mx3` 회귀 스위트.

### 회귀 스위트 완주 결과

`test/` 전체(176개 파일)를 `-failfast` 없이 완주시켰다. **panic·fatal 0건**,
실패 assertion **4건**, 그리고 **4건 전부 착수 시점 바이너리에서도 실패한다.**

| 테스트 | 값 | 판정 |
|---|---|---|
| `gammaLL.mx3` | `-0.28648561239242554` (기대 `-0.286423 ± 1e-5`) | base와 **바이트 동일** |
| `repeat.mx3` | `0.5112215058225714` (기대 `0.525 ± 1e-2`) | base와 **바이트 동일** |
| `rk4temperature.mx3` | `0.0029920609667897224` (기대 `-0 ± 1e-3`) | base와 **바이트 동일** |
| `wallclocktime.mx3` | 0.1115 s (기대 `0.105 ± 0.005`) | base 0.1166 s — **양쪽 실패, 현재가 더 근접** |

`-vet`은 176개 파일 전부 OK.

**`gammaLL.mx3`**: `alpha = 0.001`(거의 무감쇠 세차운동)에 `maxerr = 1e-4`인데
`TOL = 1e-5`, 즉 적분기 자체 오차보다 10배 엄격하다. maxerr를 조이면 기준값에서
*멀어진다*(1e-4 → 6.3e-5, 1e-5 → 3.6e-4, 1e-6 → 8.8e-4, 1e-7 → 1.2e-2). 기준값은
"CUDA가 maxerr=1e-4에서 낸 답"이고 스텝 시퀀스가 비트 단위로 일치해야만 재현된다.
백엔드가 다르면 성립할 수 없는 테스트다. maxerr 1e-5·1e-6에서도 base와 현재가 완전 동일.

**`wallclocktime.mx3`**: relax()에 0.1초 예산을 주고 **초과분이 10 ms 이내**인지 본다.
relax()는 N=3 스텝마다 벽시계를 확인하므로 granularity가 3 스텝이다. 이 M4에서 256²
한 스텝이 ~7 ms → granule ~21 ms > 허용 10 ms. 파일 자신이 "RTX 3080 mobile에서 측정,
아주 약하거나 강한 GPU도 통과하도록 여유가 충분해야 한다"고 적어놨는데 그 여유가
부족하다. 스텝이 3.3 ms를 넘는 GPU는 구조적으로 실패한다. 최적화로 스텝이 빨라져
초과분이 0.1166 → 0.1115 s로 **줄었다**(방향 일치).

**`repeat.mx3`**는 `Shape.repeat()` 지오메트리, **`rk4temperature.mx3`**는 유한온도
평균 — 둘 다 base와 값이 바이트 단위로 같다.

### 표준문제 4: 레포 기준값과의 일치

`test/standardproblem4.go`가 단언하는 기준값과 비교:

| | 기준값 | 실측 | 편차 |
|---|---|---|---|
| mx | -0.9846124053001404 | -0.9846122 | 2.1e-07 |
| my | 0.1260408908128738 | 0.1260420 | 1.2e-06 |
| mz | 0.0432712435722351 | 0.0432699 | 1.4e-06 |

허용치 1e-3 대비 약 700배 여유. 미소자성 정전 벤치마크가 이 정도로 맞는 것이
물리 정합성의 가장 강한 근거다.

### 경로별 추가 검증

벤치마크가 2D 위주였던 공백을 메웠다. Phase 2a의 버퍼 공유는 3D에서 동작이 다르다
(`fftRBuf[Z]`가 별도 버퍼).

| 경로 | 커버 대상 | 결과 |
|---|---|---|
| 3D 64×32×8 + 이방성 + DMI | `exec3D`, 별도 Z 버퍼 | 스냅샷 5/5 동일, table 1.0e-7 |
| PBC (2,2,0) | 패딩이 전혀 없는 경우 | 스냅샷 3/3 동일, table 4.0e-8 |
| Temp = 300 K | Philox RNG 브리지, B_therm | 스냅샷 3/3 동일, table **정확히 0** |
| backward Euler (solver -1) | 암시적 반복 솔버 | 스냅샷 동일, 1.37× 빠름 |

### 커밋 1의 1-ulp 합 변화 추적

`m.average()`는 `Dot`(합 리덕션)을 쓰므로 커밋 1에서 1 ulp 바뀐다(문서화된 의도).
`p05`에서 나타난 뒤 `p1c`·`p2cg`·최종까지 **값이 계속 동일** — 이후 어떤 커밋도 추가
변화를 만들지 않았다. 그 와중에도 필드 데이터(.ovf)는 계속 바이트 동일하다.

### 발견된 선행 결함: `conv_selftest.go`

`-paranoid`로만 켜지는 컨볼루션 자체검사가 **착수 시점 바이너리에서도 동일하게 실패**한다
(오차 0.4591222, 허용치 1e-6). `copyPadMul`은 `MU0*Msat`을 곱하는데 `bruteConv`는 맨 커널을
쓴다. `Msat = 1/MU0`으로 스케일을 맞추면 오차가 0.445로 *거의 그대로*라, MU0 외에 구조적
불일치가 더 있다. 2D에서 max|gpu|와 max|brute|는 0.33059 vs 0.33063으로 거의 같은데
best-fit 비가 0.57 — **크기는 맞고 위치가 틀리다.**

`test/run.bash`가 `-paranoid=false`를 명시적으로 넘기므로 이 검사는 원래부터 돌지 않았다.
Metal 포팅 이전 문제이며, 반자장 자체는 위 해석적 검증으로 정확함이 확인됐다.
**미해결로 남겼고, 되돌려 원래 상태를 유지했다.**

---

## 5. 착수하지 않은 항목과 이유

정직하게 남긴다.

### 5.1 측정으로 사망한 가설

| 가설 | 결과 |
|---|---|
| `MPSGraphExecutable` 사전 컴파일 | **1.00×** (MPSGraph가 이미 내부 캐싱) |
| rank-3 → rank-2 (길이 1 축 제거) | **1.00×** |
| `StorageModePrivate` 전환 | **0.99×** |
| `float4` 벡터화 | **0.94×** |
| blit `fillBuffer` vs compute fill | **0.90×** |
| metallib 사전 컴파일 | front-end 컴파일이 3.5 ms뿐 |
| 타일 전치로 y축 contiguous화 | 전치 59.8 µs > 절감 10.7 µs |

### 5.2 커널 융합 (Phase 3): 설계했으나 미착수

후보와 예상 이득(mesh 512² 기준):

| 융합 | 절감 | 예상 |
|---|---|---|
| `madd*` + `normalize` (step당 6회) | 6 디스패치 + m 6회 RMW | ~2.2% |
| exchange + B_ext | 3성분 RMW 1회 | ~2.2% |
| 3성분 copyPad/UnPad | eval당 4 디스패치 | 소형 메시에서 ~5% |

**미착수 이유**: 각 항목의 예상 이득(~2%)이 이 장비의 측정 노이즈(~30%)보다 훨씬 작아
**개선을 입증할 수 없는데**, 감사된 CUDA 커널 코퍼스에 새 커널 5~6개를 추가해야 한다.
`normalize`의 geometry 마스크·`is0` 처리를 정확히 복제해야 하므로 미묘한 물리 발산 위험이 실재한다.
입증 불가능한 이득에 검증 불가능한 위험을 얹는 거래여서 보류했다.

**착수 조건**: 노이즈가 작은 장비(발열 여유 있는 M-Max/Ultra) 또는 Metal System Trace로
per-dispatch 시간을 직접 볼 수 있는 환경이 있으면 위 3개를 순서대로 진행하면 된다.

### 5.3 Phase 2b 배치 FFT: 미착수

3성분을 1회 호출로 묶으면 MPSGraph encode가 step당 36 → 12로 줄고, 격리 측정에서
padded ≤256²에서 1.24~1.41×였다. 다만 `fftRBuf`/`fftCBuf`를 연속 3-plane 할당으로
바꿔야 하고(현재 `NewSlice`는 성분별 개별 할당), 2D 경로는 성분이 시간축으로 분리돼 있어
X·Y만 묶인다. 예상 end-to-end 1.08~1.13× — 역시 노이즈 이하.

### 5.4 Phase 4 ICB / concurrent dispatch: 미착수

ICB compute가 M4에서 동작하고 1577 ns/dispatch(4.34×)임을 확인했다. 하지만
- ICB는 `setBytes`를 못 써 모든 스칼라를 버퍼로 옮겨야 하고(시간 의존 파라미터는 GPU 쓰기)
- ICB 내부에 배리어를 넣을 수 없어 의존성 경계마다 `executeCommandsInBuffer`를 쪼개야 하며
- 3성분 FFT를 concurrent로 돌리려면 reduction의 슬롯 분리 가정을 재검토해야 한다

런타임 구조 변경 규모가 크고 회귀 위험이 앞의 7개 커밋을 합친 것보다 크다. 별도 작업으로 분리했다.

---

## 부록: 참고 자료

- [VkFFT — Metal 백엔드 FFT 라이브러리](https://github.com/DTolm/VkFFT) / [IEEE 논문](https://ieeexplore.ieee.org/abstract/document/10036080/)
- [From 8 Seconds to 370 ms: Kernel-Fused SAR Imaging on Apple Silicon](https://arxiv.org/html/2604.03585)
- [Kernel Fusion — single-dispatch fusion, 92 devices](https://kernelfusion.dev/)
- [Apple: Encoding indirect command buffers on the CPU](https://developer.apple.com/documentation/Metal/encoding-indirect-command-buffers-on-the-cpu)
- [Optimize machine learning for Metal apps — WWDC23](https://developer.apple.com/videos/play/wwdc2023/10050/)
- [Rigel: Metal 4.1 Tensor Compute Path on M4 Max](https://arxiv.org/html/2606.12765)
- [magnum.np — PyTorch 기반 미소자성 프레임워크](https://www.nature.com/articles/s41598-023-39192-5)
