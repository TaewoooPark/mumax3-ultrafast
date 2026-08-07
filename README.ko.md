<!-- markdownlint-disable MD033 -->

# mumax³ ultrafast

**맥에서 가장 빠른 마이크로마그네틱 시뮬레이터입니다.**

<p align="center">
  <a href="./README.md">English</a> ·
  <b>한국어</b> ·
  <a href="./README.zh.md">中文</a> ·
  <a href="./README.ja.md">日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/TaewoooPark/mumax3-ultrafast?style=flat-square&logo=github&logoColor=white&labelColor=000000&color=333333" alt="GitHub stars">
  <img src="https://img.shields.io/github/last-commit/TaewoooPark/mumax3-ultrafast?style=flat-square&labelColor=000000&color=333333" alt="Last commit">
  <img src="https://img.shields.io/github/languages/top/TaewoooPark/mumax3-ultrafast?style=flat-square&labelColor=000000&color=333333" alt="Top language">
  &nbsp;
  <img src="https://img.shields.io/badge/Apple%20Silicon-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Metal-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Metal">
  <img src="https://img.shields.io/badge/Go-000000?style=flat-square&logo=go&logoColor=white&labelColor=000000" alt="Go">
  <img src="https://img.shields.io/badge/Objective--C++-000000?style=flat-square&labelColor=000000&color=000000" alt="Objective-C++">
  &nbsp;
  <img src="https://img.shields.io/badge/mumax³%20compatible-000000?style=flat-square&labelColor=000000&color=000000" alt="mumax3 compatible">
  <img src="https://img.shields.io/badge/15%2F15%20upstream%20tests-000000?style=flat-square&labelColor=000000&color=000000" alt="15/15 upstream tests">
  <img src="https://img.shields.io/badge/No%20NVIDIA%20required-000000?style=flat-square&labelColor=000000&color=000000" alt="No NVIDIA required">
  <img src="https://img.shields.io/badge/GPL--3.0--or--later-000000?style=flat-square&labelColor=000000&color=000000" alt="GPL-3.0-or-later">
</p>

원본 [mumax³](https://github.com/mumax/3)는 NVIDIA CUDA를 전제로 만들어졌기 때문에 맥에서는 아예 실행되지 않습니다. 이 프로젝트는 GPU 실행 계층을 네이티브 **Metal** 구현으로 교체한 뒤 최적화한 결과물입니다. M4 MacBook Air 기준으로, 맥에서 돌아가는 다른 모든 마이크로마그네틱 시뮬레이터보다 **3.3배에서 30배 빠르며**, 같은 물리 계산에 **5.9배에서 26배 적은 에너지**를 씁니다.

이 포팅이 바꾼 것은 하드웨어 백엔드이지 물리 모델이 아닙니다. `.mx3` 언어, 상위 Go 솔버, 물질 항, 적분법, 출력 형식은 모두 mumax³와 호환됩니다. 결과 역시 CUDA 시절 레퍼런스와 원본 허용오차 안에서 일치합니다.

> *"가장 빠르고, 가장 효율적이며, GPU를 실제로 쓰는 유일한 시뮬레이터."*

[**taewoopark.com** 제작자 사이트](https://taewoopark.com)

<p align="center">
  <img src="./docs/bench/hero.svg" alt="애플 실리콘에서 실행되는 모든 마이크로마그네틱 시뮬레이터의 3분할 비교. M4 MacBook Air 한 대에서 측정." width="100%">
</p>

<p align="center">
  <img src="./docs/app/result-viewer-close-up.png" alt="선택형 mumax3-ultrafast 데스크톱 앱에서 OVF 벡터장을 대화형 3차원 화살표로 표시한 화면." width="100%"><br>
  <sub>선택형 데스크톱 앱에서 완료된 시뮬레이션을 바로 열어 OVF 결과를 3D로 회전·확대·색상 변경·재생할 수 있습니다.</sub>
</p>

---

## 왜 만들었는가

맥을 쓰는 물리학자에게는 지금까지 세 가지 선택지밖에 없었습니다. NVIDIA 카드를 꽂은 리눅스 장비를 따로 두거나, 빌리거나, CPU로 돌리며 기다리는 것입니다.

- **mumax³** 와 **mumax+** 는 CUDA 전용입니다. 애플 하드웨어에서 실행되지 않습니다.
- **MicroMagnetic.jl** 은 지원 백엔드 목록에 애플 GPU를 올려두고 있습니다. 그러나 0.5.0 버전은 **애플 GPU에서 실행되지 않습니다.** 패키지가 배포하는 `examples/std4.jl`을 손대지 않고 그대로 돌리면 `Sim()`에서 실패합니다. 기본 정밀도가 `Float64`인데 애플 GPU에는 배정밀도 하드웨어가 없기 때문입니다. `set_precision(Float32)`로 이 지점을 통과시켜도 GPU 커널 안에서 다시 실패합니다. demag 없이 exchange만 쓰는 LLG 스텝에서는 `gpu_fill_kernel!`이, demag 텐서에서는 `newell_f`가 걸립니다. 모든 경로가 Metal 커널 안에서 배정밀도에 닿습니다.
- **magnum.np** 에는 애플 GPU 경로 자체가 없습니다. `magnumnp/__init__.py`가 `cuda:N` 아니면 `cpu`만 선택하고, import 시점에 `float64`를 강제하며, demag 텐서도 배정밀도로 만듭니다. MPS에 올리는 데 소스 패치 세 곳이 필요했습니다.
- **OOMMF** 는 어떤 벤더의 GPU 백엔드도 없습니다.

그래서 맥은 마이크로마그네틱 계산에서 이류 장비였습니다. 더 이상 그럴 필요가 없습니다.

---

## 측정 결과

M4 MacBook Air 한 대(GPU 10코어, 32 GB, 팬리스), 한 세션, 모든 시뮬레이터가 **동일한 문제**를 풉니다. demag + exchange, 고정 1e-13초 스텝의 Heun 적분, 메시마다 동일한 스텝 수입니다. 따라서 모든 arm이 스텝당 정확히 두 번의 유효장 계산을 수행합니다. mumax³는 `Neval = 2×steps`를, OOMMF는 `energy_calc_count = 2×steps + 1`을 보고합니다.

### 속도: 모든 메시에서, 모든 상대에게 앞섭니다

| 초당 cell-evaluation | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **1.61×10⁸** | **2.21×10⁸** | **1.36×10⁸** | **1.27×10⁸** | **1.17×10⁸** |
| OOMMF (CPU, 8스레드) | 3.17×10⁷ | 3.91×10⁷ | 4.14×10⁷ | 3.41×10⁷ | 2.85×10⁷ |
| magnum.np (MPS, 패치) | 1.11×10⁷ | 2.28×10⁷ | 1.78×10⁷ | 2.63×10⁷ | 3.02×10⁷ |
| MicroMagnetic.jl (CPU) | 1.85×10⁷ | 1.62×10⁷ | 1.29×10⁷ | 1.03×10⁷ | 8.22×10⁶ |
| magnum.np (CPU, 설치 그대로) | 7.41×10⁶ | 7.36×10⁶ | 7.36×10⁶ | 5.49×10⁶ | 4.97×10⁶ |
| MicroMagnetic.jl (Metal) | 실행 불가 | 실행 불가 | 실행 불가 | 실행 불가 | 실행 불가 |

**측정한 모든 크기에서 외부 도구 대비 3.28배에서 29.97배 빠릅니다.**

<p align="center">
  <img src="./docs/bench/speed.svg" alt="메시 크기에 대한 처리량 비교." width="100%">
</p>

<sub>MicroMagnetic.jl이 두 번 등장하는 이유는 두 백엔드의 거동이 다르기 때문입니다. CPU 백엔드는 정상 실행되어 정상적으로 측정되고, Metal 백엔드는 아예 실행되지 않습니다. magnum.np도 마찬가지로 두 번 등장합니다. `MPS`는 소스 패치 세 곳을 거친 최선의 경우이고, `CPU`는 `pip install` 그대로의 상태입니다.</sub>

### 에너지: 격차가 더 벌어집니다

<p align="center">
  <img src="./docs/bench/energy.svg" alt="동일한 시뮬레이션 하나에 드는 에너지 비교." width="100%">
</p>

이 축은 NVIDIA를 상대로는 아무도 결론 낼 수 없는 축입니다. 공개된 비교들이 측정 처리량을 *공칭 보드 TDP*로 나누는데, 카드가 실제로 TDP의 몇 퍼센트를 끌어쓰는지에 따라 답이 3배씩 움직이기 때문입니다. 질문을 한 대의 기계로 좁히면 그 가정이 통째로 사라집니다. 모든 arm이 같은 칩에서 돌았고, `powermetrics`가 실제 GPU, CPU, 합산 레일을 샘플링했습니다.

| | 실행 시간 | 합산 전력 | 에너지 | **백만 cell-evals / J** |
| --- | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **7.6 s** | 6.10 W | **46.6 J** | **22.51** |
| magnum.np (MPS, 패치) | 43.4 s | 6.38 W | 277.1 J | 3.78 |
| OOMMF (CPU, 8스레드) | 25.2 s | 16.42 W | 414.2 J | 2.53 |
| MicroMagnetic.jl (CPU) | 86.1 s | 6.51 W | 560.6 J | 1.87 |
| magnum.np (CPU, 설치 그대로) | 143.7 s | 8.47 W | 1218.1 J | 0.86 |

<sub>512² 메시, Heun 2000 스텝, 즉 모든 arm에서 1.049×10⁹ cell-evaluation입니다. 측정 중 기계의 유휴 바닥값은 합산 0.195 W, GPU 0.002 W였습니다.</sub>

OOMMF는 시간으로는 3.3배 뒤질 뿐이지만 그 일을 하는 데 **16.4 W**를 태웁니다. GPU 쪽 6.1 W와 비교하면 에너지로는 **8.9배** 뒤집니다. 팬 없는 노트북이 백열전구 필라멘트 수준의 전력으로 같은 물리를 계산합니다.

### 용량: 노트북에서 8390만 셀

<p align="center">
  <img src="./docs/bench/capacity.svg" alt="32 GB MacBook Air에 들어가는 최대 문제 크기." width="100%">
</p>

**83,886,080 셀**(8192 × 10240, 피크 23.69 GiB)이 스왑 없이 완주합니다. 4 nm 셀 기준으로 **32.8 × 41.0 µm** 박막이며, 소자의 한 귀퉁이가 아니라 패턴이 들어간 소자 전체에 해당합니다. 이 플랫폼에서 GPU를 쓰는 다른 어떤 도구보다 **5.0배 큽니다**.

OOMMF도 같은 8390만 셀에 도달합니다. 다만 그 크기에서 mumax3-ultrafast가 **4.5배 빠릅니다**(29.2초 대 131.1초). 두 도구 모두 이 기계에서 1억 490만 셀에는 실패합니다.

### 물리: 독립적으로 작성된 네 코드, 하나의 답

<p align="center">
  <img src="./docs/bench/physics.svg" alt="네 시뮬레이터의 평균 자화 상대 편차." width="100%">
</p>

답이 틀리면 속도 숫자는 무의미합니다. 위의 모든 arm은 타이밍을 집계하기 전에 먼저 일치를 통과해야 했습니다. 같은 스텝 수를 지난 뒤, 서로 독립적으로 작성된 네 코드가(세 언어, 세 백엔드) 같은 평균 자화를 내놓습니다.

| 동일 스텝 후 ⟨mₓ⟩ | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: |
| mumax3-ultrafast | 0.994939 | 0.9950321 | 0.9950429 |
| OOMMF | 0.994942 | 0.9950322 | 0.9950370 |
| MicroMagnetic.jl (CPU) | 0.994939 | 0.9950319 | 0.9950369 |
| magnum.np (MPS) | 0.994939 | 0.9950319 | 0.9950370 |

전체 스윕에서 최대 불일치는 **7.1×10⁻⁵**입니다.

---

## 물리적 정합성

방정식과 솔버를 **다시 구현하지 않았습니다**. 네이티브 백엔드는 기존 mumax³ 물리 모델을 Metal을 통해 실행할 뿐이므로, 인용하게 되는 모델은 여전히 mumax³입니다. 수정하지 않은 원본 CUDA 시절 회귀 레퍼런스에 대한 검증 결과는 다음과 같습니다.

| 검증 항목 | 결과 |
| --- | ---: |
| 통과한 비열적 원본 물리 테스트 | **15 / 15** |
| 원본 허용오차 이내 어서션 | **103 / 103 (100.000%)** |
| 평균 자화 벡터 일치도 평균 | **99.9923%** |
| 평균 자화 벡터 일치도 최솟값 | **99.9415%** |
| 허용오차 0 어서션 정확 일치 | **19 / 19** |
| 완주한 공식 mumax³ 예제 페이지 시뮬레이션 | **15 / 15** |

100%라는 수치는 허용오차 부합이지 비트 단위 동일성 주장이 아닙니다. 병렬 GPU 리덕션은 마지막 부동소수점 비트가 다를 수 있습니다. 열적 시뮬레이션은 cuRAND의 XORWOW 대신 Philox를 쓰므로, 시드는 Metal에서 재현되지만 CUDA의 샘플 단위 궤적까지 재현하지는 않습니다. 대신 통계적 거동을 검증했습니다.

공식 예제 입력 15개는 모두 원본 템플릿과 바이트 단위로 동일하며, 출력과 로그와 출처가 함께 커밋되어 있습니다.

- [전체 물리 검증 방법과 데이터](physics-validation-results/apple-m4-cuda-era-20260731/)
- [공식 예제, 출력, 출처, 라이선스](examples-and-results/RESULTS.md)

---

## 설치

지원되는 맥에서 터미널을 열고 실행합니다.

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-macos.sh)"
```

설치 스크립트는 먼저 최신 GitHub 릴리스에서 사전 빌드된 애플 실리콘 엔진과 함께 배포된 `.sha256` 파일을 찾습니다. 해당 자산이 있으면 SHA-256 체크섬을 검증하고 실행 파일을 `~/.local/bin/mumax3`에 설치한 뒤, 이 경로를 `~/.zprofile`에 추가하고 `mumax3 -test`로 마무리합니다. 이 사전 빌드 경로에는 Homebrew, Go, Git, 소스 체크아웃이 필요하지 않습니다. 이 명령은 데스크톱 앱을 설치하지 않습니다.

엔진 아카이브가 명시적인 HTTP 404를 반환하면—예를 들어 아직 자산이 배포되지 않은 버전이면—설치 스크립트가 자동으로 소스 빌드로 전환하며 Apple Command Line Tools, 네이티브 Homebrew, Go를 설치할 수 있습니다. 네트워크 및 기타 연결 오류에서는 이 전환을 하지 않고 안전하게 중단합니다. 중간에 끊긴 설치는 다시 실행해도 됩니다.

이미 체크아웃한 저장소에서 소스 빌드를 강제로 실행하려면 다음을 사용합니다.

```bash
./install-macos.sh --from-source
```

설치 후에는 새 터미널을 열어야 갱신된 경로가 반영됩니다.

> [!IMPORTANT]
> 애플 실리콘(M1 이상), macOS 14 이상이 필요합니다. 인텔 맥은 지원하지 않습니다. NVIDIA GPU가 있는 리눅스와 윈도우에서는 원본 CUDA 백엔드를 그대로 쓰면 됩니다.

### 선택 사항: 데스크톱 앱

<p align="center">
  <img src="./apps/desktop/src-tauri/icons/icon.png" alt="흰색 박스 위에 엑스트라 볼드 산세리프 m과 간격을 둔 위첨자 3을 시각적으로 중앙 정렬한 mumax3 ultrafast 데스크톱 앱 아이콘." width="112">
</p>

위에서 설치하는 `mumax3-ultrafast` 엔진과 명령줄 사용 방식이 기본이며, macOS 데스크톱 앱은 그 위에 선택적으로 더하는 인터페이스입니다. `.mx3` 파일을 기준으로 삼으면서 편집기, 작업 폴더 선택, 원클릭 실행, 실시간 자화와 솔버 수치, 실행 로그, 통합 3D OVF 결과 뷰어를 제공합니다.

Apple Developer Program 가입 없이 선택형 앱을 이 Mac에서 직접 빌드하고 설치할 수 있습니다.

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)"
```

설치기는 최신 태그 릴리스를 찾고 `~/Library/Caches/mumax3-ultrafast` 아래에 전용 Node.js·pnpm·Rust 도구를 준비한 다음, 이 Mac에서 앱을 빌드하고 로컬 ad-hoc 서명을 적용합니다. 완성된 앱은 `~/Applications/mumax3 ultrafast.app`에 저장되며 같은 버전의 독립형 엔진도 함께 설치됩니다. 첫 소스 빌드는 몇 분 걸릴 수 있지만 이후에는 전용 캐시를 재사용합니다. Apple Command Line Tools가 없다면 시스템 설치 창을 완료한 뒤 같은 명령을 다시 실행하세요. 업데이트할 때도 같은 명령을 다시 실행하면 됩니다. 엔진·빌드 캐시·시뮬레이션 파일은 보존하고 앱만 제거하려면 다음 명령을 사용합니다.

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)" -- --uninstall
```

이 로컬 소스 빌드 앱은 Apple 공증 앱이 아니므로 기관에서 관리하는 Mac에서는 제한될 수 있습니다. [GitHub Release](https://github.com/TaewoooPark/mumax3-ultrafast/releases)에 `mumax3-ultrafast-app-darwin-arm64.dmg`가 포함되어 있다면 그 자산이 더 빠른 서명·공증 설치 경로입니다. 옵션과 개발 절차는 [`apps/desktop/README.md`](apps/desktop/README.md)를 참고하세요.

<table>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/desktop-workspace-editor.png" alt="시뮬레이션 작업 공간과 mx3 편집기를 표시한 데스크톱 앱." width="100%"><br><sub><b>스크립트 작업 공간</b> — 선택한 폴더에서 <code>.mx3</code> 파일을 열거나 작성합니다.</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/desktop-simulation-running.png" alt="실시간 수치와 함께 실행 중인 시뮬레이션을 표시한 데스크톱 앱." width="100%"><br><sub><b>실시간 실행</b> — 자화, 솔버 수치, 진행률, 로그를 한눈에 확인합니다.</sub></td>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/desktop-results-ready.png" alt="완료된 시뮬레이션의 OVF 프레임을 표시한 데스크톱 앱." width="100%"><br><sub><b>결과 준비</b> — 작업 공간을 떠나지 않고 완료된 실행의 OVF 프레임을 엽니다.</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-overview.png" alt="OVF 벡터장의 대화형 3D 전체 보기." width="100%"><br><sub><b>3D 벡터장</b> — 결과를 회전·이동·확대·색상 변경·재생합니다.</sub></td>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-close-up.png" alt="3D 결과 뷰어에 표시된 색상 벡터 화살표의 확대 화면." width="100%"><br><sub><b>세부 보기</b> — 글리프 형태와 방향·크기 색상 모드를 전환합니다.</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-top-view.png" alt="데스크톱 결과 뷰어에 표시된 OVF 벡터장의 상단 보기." width="100%"><br><sub><b>상단 보기</b> — 시뮬레이션 평면의 텍스처를 살펴봅니다.</sub></td>
  </tr>
</table>

## 시뮬레이션 실행

```bash
mumax3 example.mx3            # http://127.0.0.1:35367 에서 실시간 웹 UI
mumax3 -http="" example.mx3   # 헤드리스, 벤치마크와 배치 작업용
```

출력은 `example.out/`에 생성됩니다. 두 모드는 같은 시뮬레이션을 실행하고 같은 출력을 냅니다. Xcode 전체 설치와 오프라인 Metal 컴파일러는 필요 없습니다. 셰이더 라이브러리는 시스템 Metal 런타임을 통해 컴파일됩니다.

---

## 네이티브 포팅의 구조

| 원본 CUDA 구성요소 | macOS 구현 | 유지된 호환성 |
| --- | --- | --- |
| CUDA 컴퓨트 커널 | Metal 컴퓨트 셰이더와 Objective-C++ 런타임 브리지 | 기존 mumax³ 커널 계약과 FP32 모델 |
| cuFFT | MPSGraph FFT, 조건을 만족하는 2D 변환에는 벤더링한 VkFFT 경로 | demag 합성곱이 쓰는 패킹된 R2C/C2R 레이아웃 |
| `cudaMalloc` 과 CUDA 스트림 | 애플 통합 메모리 버퍼, 커맨드 배칭, 순서가 보장된 단일 Metal 큐 | 버퍼 의미론과 실행 순서 |
| cuRAND 열적 노이즈 | Philox4x32-10과 Box–Muller 정규분포 생성 | Metal에서의 시드 재현성, 검증된 노이즈 통계 |
| CUDA 전용 플랫폼 바인딩 | `darwin/arm64` 빌드 태그와 호환 심 | `.mx3` 언어와 상위 Go API |

구현은 애플 GPU에 맞춘 X 연속 32폭 SIMD 타일을 쓰고, 블로킹 드레인 사이에도 GPU를 상주시키며, MPSGraph의 스케줄링이 무너지는 구간을 피하려고 큰 demag 변환을 축 단위로 분할하고, 셰이더를 런타임에 컴파일해서 Command Line Tools만으로 충분하게 만듭니다.

---

## 튜닝

### 레이턴시 바운드 실행

연구용 메시 크기에서 애플 GPU는 연산이 부족하기보다 호스트를 기다리며 노는 경우가 많습니다. 128×128에서는 호스트가 Dormand-Prince 스텝을 인코딩하는 시간이 GPU가 그 스텝을 실행하는 시간보다 깁니다. 이를 겨냥한 스위치가 셋 있으며, 어느 것도 기본 빌드의 동작을 바꾸지 않습니다.

```go
SpeculativeStep = true   // 호스트 인코딩과 GPU 실행을 겹칩니다. 실측 1.53배
MinimizeOnGPU   = true   // minimize()의 BB 스텝 크기를 디바이스에 둡니다. 실측 1.58배
```

```sh
mumax3 -j 3 sweep_*.mx3   # GPU당 입력 N개를 큐잉합니다. minimize() 3개 기준 합산 2.53배
```

`SpeculativeStep`은 `MaxErr`를 계속 강제하지만 기각이 한 스텝 늦게 일어납니다. 그래서 시간 스텝 수열이 정확한 제어기와 달라지고, 수천 스텝이 지나면 궤적이 약 1% 수준으로 갈라집니다. `FixDt`, 유한한 `Temp`, `relax()`, `DemagExtrapolation`, 사후 훅 아래에서는 스스로 닫힙니다. `MinimizeOnGPU`는 모든 하강을 비트 단위로 동일하게 유지하고, 수렴 판정만 한 반복 늦어지므로 최소화가 한 반복 더 진행된 지점에서 멈출 수 있습니다. 둘 다 각자의 문제에 대해 기본 실행과 비교 검증하고 쓰시는 것이 좋습니다.

### Metal FFT 백엔드

기본값은 패딩된 크기가 512×512 이하의 2의 거듭제곱이고 활성 데이터가 엄격한 접두인 2D demag 변환에 대해 벤더링한 VkFFT 백엔드를 자동으로 쓰고, 나머지는 MPSGraph가 처리합니다. `MUMAX3_METAL_FFT_BACKEND=mps`는 모든 플랜을 MPSGraph에 고정하고, `vkfft`는 패딩된 1024×1024 구간까지 VkFFT를 확장합니다.

### 선택적 demag 외삽

고차 반자기장 외삽은 demag 비중이 큰 솔버 4/5/6 작업을 가속하지만 근사이므로 **기본적으로 꺼져 있습니다.**

```go
SetSolver(5)
DemagExtrapolation = true
```

지원되지 않는 솔버와 안전하지 않은 모델 상태에서는 정확한 합성곱으로 닫히며 실패합니다. 오차는 궤적과 시간 스텝에 따라 달라지므로 다른 문제에서의 성공적인 벤치마크가 정확도를 보장하지 않습니다. [검증 결과](bench/demag-extrap/RESULTS.md)와 [A/B 방법](bench/demag-extrap/README.md)을 먼저 확인하시는 것이 좋습니다.

### 크기 선택

셀당 처리량은 문제 크기에 대해 평탄하지 않습니다. 측정한 M4에서는 256²에서 정점을 찍고, 128² 이하에서는 약 172~187 µs의 고정 계산당 오버헤드가 지배하므로 GPU가 넓어져도 전혀 도움이 되지 않습니다. 측정한 곡선은 `bench/curve.txt`에 있습니다.

---

## 벤치마크 재현

위의 모든 내용은 이 저장소에서 재현 가능합니다. 모델링한 값은 하나도 없습니다.

```bash
./bench/capacity.sh name=/path/to/binary   # Metal 한계, 셀당 바이트, vmmap 분해, 천장
python3 bench/crosstool_svg.py             # 측정 데이터로 README 차트 재생성
```

시뮬레이터마다 하나씩, 모두 같은 문제를 같은 적분기로 푸는 교차 도구 하네스는 [`crosstool/`](crosstool/)에 있습니다. `run_crosstool.py`가 속도와 물리를, `run_energy.py`가 powermetrics를, `run_capacity.py`가 천장을 담당합니다. magnum.np를 애플 GPU에 올리는 데 필요한 패치와, MicroMagnetic.jl의 Metal 백엔드가 실행 불가라고 결론짓기까지 시도한 내용은 각 스크립트의 독스트링에 적어 두었습니다.

결과를 실제로 바꾼 계측상의 주의점 두 가지를 남깁니다. 스왑은 `Pageouts`가 아니라 `vm_stat`의 **`Swapouts`**로 감시해야 합니다. 애플 실리콘의 압축기는 기가바이트 단위를 쓰면서도 `Pageouts`를 거의 움직이지 않습니다. 그리고 mumax³의 기하별 demag 커널 캐시는 이 크기대에서 항목 하나가 8~10 GB에 달하므로 실행 사이에 반드시 지워야 합니다.

---

## 주장의 범위

한계를 감추는 벤치마크는 가치가 없으므로 그대로 적습니다.

- **NVIDIA를 상대로 절대 속도로는 집니다.** mumax³ 자체의 62개 GPU 벤치마크 표에서 M4는 GTX 1650 mobile과 GTX 970 사이에 놓입니다. 여기서의 주장을 애플 실리콘으로 한정한 것은 의도적입니다. 모든 축을 한 기계에서, TDP 가정 없이, 기계 간 정규화 없이 측정할 수 있는 범위가 그것이기 때문입니다.
- **용량은 OOMMF와 동률입니다.** 둘 다 8390만 셀에 도달하고 둘 다 1억 490만 셀에서 실패합니다. mumax3-ultrafast는 그 크기에서 4.5배 빠를 뿐, 더 크지는 않습니다.
- **직전 릴리스 대비 격차는 메시가 커질수록 좁아집니다.** 128²에서 5.4배, 2048²에서 1.38배입니다. 이번 작업이 겨냥한 것은 연구용 메시가 실제로 놓이는 레이턴시 바운드 구간이었습니다.

---

## 원본과 라이선스

이 프로젝트는 [mumax³](https://github.com/mumax/3)에서 파생했으며, 그 설계와 검증은 [원 논문](https://doi.org/10.1063/1.4899186)에 기술되어 있습니다. NVIDIA CUDA 사용자는 [공식 mumax³ 설치 문서](https://mumax.github.io/download.html)를 따르시면 됩니다.

[GNU GPL v3 이상](LICENSE)으로 배포합니다. [프로젝트 고지](NOTICE)는 원본의 CUDA 링킹 허가를 유지하고 macOS 수정 사항을 명시합니다. Random123, Go, SVGo, Freetype-Go에 대한 라이선스와 저작자 표시는 [서드파티 고지](THIRD_PARTY_NOTICES.md)에 모아 두었습니다.

비교 대상은 측정 당시 설치된 버전으로 표기합니다. [MicroMagnetic.jl](https://github.com/ww1g11/MicroMagnetic.jl) 0.5.0과 Metal.jl 1.10([arXiv:2406.16064](https://arxiv.org/abs/2406.16064)), [magnum.np](https://gitlab.com/magnum.np/magnum.np) 2.2.0, [OOMMF](https://math.nist.gov/oommf/) 2.0b0입니다.

---

## 제작자

저는 **박태우**이고, 한국과학기술원(KAIST) 물리학과 학부생입니다. 2025년 10월부터 **김갑진 교수**가 이끄는 [KAIST 초고속 스핀 동역학 연구실(USDL)](https://spintronics.kaist.ac.kr/)에서 자성 도메인월 운동과 뉴로모픽 컴퓨팅 응용에 관한 실험 스핀트로닉스 연구를 수행하고 있습니다. 2023년 6월부터 2024년 3월까지는 **김세권 교수**의 KAIST 양자 스핀 동역학 연구실에서 이론 모델링과 마이크로마그네틱 시뮬레이션으로 도메인월 운동을 연구했습니다.

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>같은 방정식, 같은 스크립트, 같은 답. 애플 실리콘에서.</sub></p>
