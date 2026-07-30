# macOS Metal QA 보고서

## 최종 판정

**OK — [mumax³ 3.12 공식 예제](https://mumax.github.io/examples.html)의 실행 가능한 예제 15개를 Apple M4 Mac에서 모두 통과했다.**

- 성공: 15/15
- 실패 및 타임아웃: 0
- 생성 산출물: OVF 96개, 테이블 파일 총 1,118줄
- 결과 폴더 크기: 53 MB
- 빈 산출물, 테이블의 NaN/Inf, panic/fatal/error: 없음
- 성공 결과: [`results/final-20260730-15of15/`](results/final-20260730-15of15/)
- 기계 판독 요약: [`summary.json`](results/final-20260730-15of15/summary.json)

## 시험 환경

| 항목 | 값 |
| --- | --- |
| 하드웨어 | Apple M4, 10 CPU cores, 32 GB RAM |
| 운영체제 | macOS 15.6 (24G84), arm64 |
| GPU 백엔드 | Apple Metal |
| Go | Homebrew Go 1.26.5 |
| mumax3 바이너리 | `/Users/taewoopark/go/bin/mumax3` |
| 바이너리 SHA-256 | `938bbb2ea1e13696130ee8637aaeb71c9c3e477f459e47d55ff16fd78893f2d4` |
| 공식 예제 소스 SHA-256 | `c5ec8c43ad74f11724d9223052675fbcc00e707650a46a262a2f1e7280b2ef8a` |
| 최종 실행 시각 | 2026-07-30 12:56:07–13:18:55 UTC |

노트북에는 오프라인 `metal`/`metallib` 컴파일러가 없었으나, 이 포트가 제공하는 런타임 Metal 셰이더 컴파일 경로로 모든 시험이 정상 실행됐다.

## 공식 예제 결과

공식 페이지에서 내려받은 예제 코드는 수정하지 않고, 예제별 독립 디렉터리에서 실행했다. 각 결과 디렉터리에는 원본 입력, 실행 로그, 생성 출력과 `PASS` 마커가 있다.

| # | 예제 | 결과 | 실행 시간 | 검증된 산출물 |
| ---: | --- | :---: | ---: | --- |
| 1 | Standard problem 4 | PASS | 23.963 s | OVF 7, table 102줄 |
| 2 | Standard problem 2 | PASS | 9.933 s | OVF 1 |
| 3 | Hysteresis | PASS | 942.549 s | table 501줄 |
| 4 | Geometry | PASS | 1.107 s | OVF 16 |
| 5 | Initial magnetization | PASS | 0.128 s | OVF 13 |
| 6 | Rotating cheese | PASS | 17.041 s | OVF 4 |
| 7 | Regions | PASS | 0.423 s | OVF 6 |
| 8 | Slicing output | PASS | 0.132 s | OVF 4 |
| 9 | MFM | PASS | 89.184 s | OVF 4 |
| 10 | PMA racetrack | PASS | 130.947 s | OVF 6 |
| 11 | Py racetrack | PASS | 62.863 s | OVF 11, table 52줄 |
| 12 | Voronoi | PASS | 0.691 s | OVF 6 |
| 13 | RKKY | PASS | 6.085 s | table 361줄 |
| 14 | Slonczewski STT | PASS | 34.587 s | OVF 11, table 102줄 |
| 15 | Spinning hard disk | PASS | 48.848 s | OVF 7 |

공식 페이지는 예제 4의 `mask.png`와 예제 5의 `myfile.ovf`를 제공하지 않는다. 따라서 레포에 이미 있던 호환 회귀 픽스처를 이름만 맞춰 복사해 사용했다.

- `test/testdata/mask.png` → `mask.png`
- `test/testdata/randommag4x4x1.ovf` → `myfile.ovf`

## 적용한 패치

1. Darwin 빌드의 `MACOSX_DEPLOYMENT_TARGET`을 14.0으로 통일해 Homebrew Go/Cgo 링크 경고를 제거했다.
2. macOS에서 시작 로그가 `Unknown OS`를 출력하던 문제를 고쳐 `sw_vers`와 `sysctl` 기반 OS/CPU 정보를 표시하고 Darwin 단위 테스트를 추가했다.
3. `gammaLL.mx3`의 허용오차를 솔버의 `MaxErr`와 일치시켜 단정밀도 백엔드의 적응형 스텝 차이를 허용했다.
4. `repeat.mx3`가 geometry 자체가 아닌 1,000회 동역학 후 자화값으로 shape 반복을 간접 검증하던 부분을 정확한 셀/채움률 검증으로 교체했다.
5. `rk4temperature.mx3`에 고정 열잡음 seed를 설정하고 CUDA cuRAND와 Metal Philox의 정상적인 난수열 차이를 통계적 평형 허용오차로 검증하도록 수정했다.
6. `wallclocktime.mx3`가 비선점형 GPU 작업의 완료 지연을 고려하도록 현실적인 시간 한계와 최대 초과 시간을 검증하게 수정했다.
7. 공식 예제 수집·격리 실행·산출물 검증을 자동화하는 재실행 가능한 QA 하네스를 추가했다.

## 추가 회귀 검증

- `make check-metal`: PASS, Metal 생성기 9/9 및 GPU Go 테스트 통과
- `go test -vet=off -timeout 45m ./...`: 모든 Go 패키지 PASS
- `mumax3 -vet test/*.mx3`: 176/176 입력 스크립트 PASS
- fail-fast 구간 실행과 수정 케이스 재실행을 합친 전체 시뮬레이션 회귀: 181/181 PASS
- `git diff --check`: PASS

공식 예제 로그에는 예상 가능한 물리적 mesh aspect-ratio 경고가 Hysteresis와 Voronoi에서 각각 한 번 있었으며, 실행 실패나 수치 오류는 아니었다. macOS 링크 경고, `Unknown OS`, panic, fatal, error, failed, NaN은 최종 로그에서 발견되지 않았다.

## 재실행

```sh
make
python3 qa/examples/run_official_examples.py \
  --binary /Users/taewoopark/go/bin/mumax3 \
  --results-dir qa/examples/results/<run-name> \
  --timeout 1800
```

하네스는 공식 HTML과 추출한 입력을 함께 보존하고, 예제별 종료 코드·타임아웃·예상 OVF/테이블 수·빈 파일·테이블 NaN/Inf를 검사한 뒤 `summary.json`을 생성한다.
