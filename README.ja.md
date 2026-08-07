<!-- markdownlint-disable MD033 -->

# mumax³ ultrafast

**Mac で最速のマイクロマグネティクスシミュレータです。**

<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README.ko.md">한국어</a> ·
  <a href="./README.zh.md">中文</a> ·
  <b>日本語</b>
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

本家の [mumax³](https://github.com/mumax/3) は NVIDIA CUDA を前提に作られているため、Mac ではそもそも動作しません。本プロジェクトは GPU 実行層をネイティブな **Metal** 実装に置き換え、そのうえで最適化したものです。M4 MacBook Air において、Mac で動作する他のあらゆるマイクロマグネティクスシミュレータより **3.3 倍から 30 倍高速**であり、同じ物理計算を **5.9 分の 1 から 26 分の 1 のエネルギー**で実行します。

この移植が変えたのはハードウェアバックエンドであって、物理モデルではありません。`.mx3` 言語、上位の Go ソルバ、材料項、積分法、出力形式はいずれも mumax³ 互換のままです。計算結果も、本家が定めた許容誤差の範囲内で CUDA 時代のリファレンスと一致します。

> *「最速。最も省電力。そして GPU を実際に使っている唯一のもの。」*

[**taewoopark.com** 作者サイト](https://taewoopark.com)

<p align="center">
  <img src="./docs/bench/hero.svg" alt="Apple Silicon で動作するすべてのマイクロマグネティクスシミュレータを、1 台の M4 MacBook Air で測定した 3 パネル比較。" width="100%">
</p>

<p align="center">
  <img src="./docs/app/result-viewer-close-up.png" alt="任意導入の mumax3-ultrafast デスクトップアプリで、OVF ベクトル場を対話的な 3 次元矢印として表示した画面。" width="100%"><br>
  <sub>任意導入のデスクトップアプリで完了したシミュレーションを直接開き、OVF 結果を 3D で回転、ズーム、色変更、再生できます。</sub>
</p>

---

## なぜ作ったか

Mac を使う物理研究者にはこれまで三つの選択肢しかありませんでした。NVIDIA カードを積んだ Linux 機を別に用意するか、借りるか、CPU で回して待つかです。

- **mumax³** と **mumax+** は CUDA 専用で、Apple のハードウェアでは動作しません。
- **MicroMagnetic.jl** は対応バックエンドに Apple GPU を挙げていますが、0.5.0 は **Apple GPU では動作しません**。同梱の `examples/std4.jl` を無修正で実行すると `Sim()` で失敗します。既定の精度が `Float64` である一方、Apple GPU には倍精度のハードウェアが存在しないためです。`set_precision(Float32)` でここを通しても、今度は GPU カーネル内部で失敗します。反磁場を含まず交換相互作用だけの LLG ステップでは `gpu_fill_kernel!` が、反磁場テンソルでは `newell_f` が引っかかります。あらゆる経路が Metal カーネル内で倍精度に到達します。
- **magnum.np** には Apple GPU 経路そのものがありません。`magnumnp/__init__.py` は `cuda:N` か `cpu` しか選ばず、import 時に `float64` を強制し、反磁場テンソルも倍精度で構築します。MPS に載せるにはソースへの三箇所の修正が必要でした。
- **OOMMF** はどのベンダーの GPU バックエンドも持ちません。

そのため Mac はマイクロマグネティクス計算において二級の機械でした。もうその必要はありません。

---

## 測定結果

M4 MacBook Air 1 台（GPU 10 コア、32 GB、ファンレス）、1 セッション、すべてのシミュレータが**同一の問題**を解きます。反磁場 + 交換相互作用、固定 1e-13 秒ステップの Heun 積分、メッシュごとに同一のステップ数です。したがって各アームはステップあたり厳密に 2 回の有効磁場計算を実行します。mumax³ は `Neval = 2×steps` を、OOMMF は `energy_calc_count = 2×steps + 1` を報告します。

### 速度：すべてのメッシュで、すべての相手に対して先行

| 毎秒 cell-evaluation | 128² | 256² | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **1.61×10⁸** | **2.21×10⁸** | **1.36×10⁸** | **1.27×10⁸** | **1.17×10⁸** |
| OOMMF（CPU、8 スレッド） | 3.17×10⁷ | 3.91×10⁷ | 4.14×10⁷ | 3.41×10⁷ | 2.85×10⁷ |
| magnum.np（MPS、パッチ適用） | 1.11×10⁷ | 2.28×10⁷ | 1.78×10⁷ | 2.63×10⁷ | 3.02×10⁷ |
| MicroMagnetic.jl（CPU） | 1.85×10⁷ | 1.62×10⁷ | 1.29×10⁷ | 1.03×10⁷ | 8.22×10⁶ |
| magnum.np（CPU、素のインストール） | 7.41×10⁶ | 7.36×10⁶ | 7.36×10⁶ | 5.49×10⁶ | 4.97×10⁶ |
| MicroMagnetic.jl（Metal） | 実行不可 | 実行不可 | 実行不可 | 実行不可 | 実行不可 |

**測定したすべてのサイズで、外部ツールに対し 3.28 倍から 29.97 倍高速です。**

<p align="center">
  <img src="./docs/bench/speed.svg" alt="メッシュサイズに対するスループット。" width="100%">
</p>

<sub>MicroMagnetic.jl が 2 回登場するのは、2 つのバックエンドの挙動が異なるためです。CPU バックエンドは正常に動作して通常どおり測定できますが、Metal バックエンドはまったく動作しません。magnum.np も同様に 2 回登場します。`MPS` はソース 3 箇所の修正を経た最良のケース、`CPU` は `pip install` そのままの状態です。</sub>

### エネルギー：差はさらに開きます

<p align="center">
  <img src="./docs/bench/energy.svg" alt="同一のシミュレーション 1 回に要するエネルギーの比較。" width="100%">
</p>

この軸は NVIDIA を相手に誰も決着をつけられない軸です。公表されている比較は実測スループットを*公称ボード TDP* で割っており、カードが実際に TDP の何割を消費するかによって答えが 3 倍変わるからです。問いを 1 台の機械に限れば、その仮定は丸ごと消えます。すべてのアームが同一のチップ上で動作し、`powermetrics` が実際の GPU、CPU、合計レールをサンプリングしました。

| | 実時間 | 合計電力 | エネルギー | **百万 cell-evals / J** |
| --- | ---: | ---: | ---: | ---: |
| **mumax3-ultrafast** | **7.6 s** | 6.10 W | **46.6 J** | **22.51** |
| magnum.np（MPS、パッチ適用） | 43.4 s | 6.38 W | 277.1 J | 3.78 |
| OOMMF（CPU、8 スレッド） | 25.2 s | 16.42 W | 414.2 J | 2.53 |
| MicroMagnetic.jl（CPU） | 86.1 s | 6.51 W | 560.6 J | 1.87 |
| magnum.np（CPU、素のインストール） | 143.7 s | 8.47 W | 1218.1 J | 0.86 |

<sub>512² メッシュ、Heun 2000 ステップ、すなわち各アームで 1.049×10⁹ 回の cell-evaluation です。測定中の機械のアイドル値は合計 0.195 W、GPU 0.002 W でした。</sub>

OOMMF は時間では 3.3 倍遅れるだけですが、その仕事に **16.4 W** を費やします。GPU 側の 6.1 W と比べると、エネルギーでは **8.9 倍**の差になります。ファンのないノートパソコンが、白熱電球のフィラメント並みの電力で同じ物理を計算しています。

### 容量：ノートパソコンで 8390 万セル

<p align="center">
  <img src="./docs/bench/capacity.svg" alt="32 GB MacBook Air に収まる最大の問題規模。" width="100%">
</p>

**83,886,080 セル**（8192 × 10240、ピーク 23.69 GiB）がページアウトなしで完走します。4 nm セル換算で **32.8 × 41.0 µm** の薄膜であり、素子の一角ではなくパターン化された素子全体に相当します。これはこのプラットフォームで GPU を使える他のどのツールよりも **5.0 倍大きい**規模です。

OOMMF も同じ 8390 万に到達します。ただしそのサイズにおいて mumax3-ultrafast は **4.5 倍高速**です（29.2 秒に対し 131.1 秒）。両者ともこの機械では 1 億 490 万には失敗します。

### 物理：独立に書かれた 4 つのコード、1 つの答え

<p align="center">
  <img src="./docs/bench/physics.svg" alt="4 つのシミュレータにおける平均磁化の相対偏差。" width="100%">
</p>

答えが誤っていれば速度の数字に意味はありません。上記のすべてのアームは、計時が採用される前にまず一致検証を通過する必要がありました。同じステップ数を経たのち、独立に書かれた 4 つのコード（3 言語、3 バックエンド）が同じ平均磁化を返します。

| 同一ステップ後の ⟨mₓ⟩ | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: |
| mumax3-ultrafast | 0.994939 | 0.9950321 | 0.9950429 |
| OOMMF | 0.994942 | 0.9950322 | 0.9950370 |
| MicroMagnetic.jl（CPU） | 0.994939 | 0.9950319 | 0.9950369 |
| magnum.np（MPS） | 0.994939 | 0.9950319 | 0.9950370 |

スイープ全体での最大の不一致は **7.1×10⁻⁵** です。

---

## 物理的整合性

方程式とソルバは**再実装していません**。ネイティブバックエンドは既存の mumax³ の物理モデルを Metal 経由で実行するだけなので、引用するモデルは依然として mumax³ です。無修正の本家 CUDA 時代の回帰リファレンスに対する検証結果は次のとおりです。

| 検証項目 | 結果 |
| --- | ---: |
| 通過した非熱的な本家物理テスト | **15 / 15** |
| 本家の許容誤差内に収まったアサーション | **103 / 103（100.000%）** |
| 平均磁化ベクトルの一致度（平均） | **99.9923%** |
| 平均磁化ベクトルの一致度（最小） | **99.9415%** |
| 許容誤差ゼロのアサーションの完全一致 | **19 / 19** |
| 完走した公式 mumax³ サンプルページのシミュレーション | **15 / 15** |

この 100% は許容誤差への適合であって、ビット単位の一致を主張するものではありません。並列 GPU リダクションは浮動小数点の下位ビットが異なりうるためです。熱的シミュレーションは cuRAND の XORWOW ではなく Philox を用いるため、シードは Metal 上で再現できますが、CUDA のサンプル単位の軌跡までは再現しません。代わりに統計的な振る舞いを検証しています。

15 個の公式サンプル入力はいずれも本家テンプレートとバイト単位で同一であり、その出力、ログ、来歴もコミットされています。

- [物理検証の方法とデータの全体](physics-validation-results/apple-m4-cuda-era-20260731/)
- [公式サンプル、出力、来歴、ライセンス](examples-and-results/RESULTS.md)

---

## インストール

対応する Mac でターミナルを開き、次を実行します。

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-macos.sh)"
```

インストーラはまず、最新の GitHub リリースにあるビルド済み Apple Silicon エンジンと公開済みの `.sha256` ファイルを探します。これらが利用できる場合は SHA-256 チェックサムを検証し、実行ファイルを `~/.local/bin/mumax3` として配置して、そのディレクトリを `~/.zprofile` に追加し、最後に `mumax3 -test` を実行します。このビルド済みバイナリの経路では、Homebrew、Go、Git、ソースのチェックアウトは不要です。このコマンドがデスクトップアプリを導入することはありません。

エンジンアーカイブが明示的な HTTP 404 を返した場合、たとえば対象バージョンのアセットがまだ公開されていない場合には、インストーラは自動的にソースビルドへ切り替わり、Apple Command Line Tools、ネイティブ Homebrew、Go を導入することがあります。ネットワークなどの接続エラーではこの切り替えを行わず、安全に停止します。途中で中断しても、再実行して問題ありません。

既存のチェックアウトからソースビルドを明示的に行うには、次を実行します。

```bash
./install-macos.sh --from-source
```

インストール後は、更新されたパスを読み込むために新しいターミナルを開いてください。

> [!IMPORTANT]
> Apple Silicon（M1 以降）と macOS 14 以降が必要です。Intel Mac には対応していません。NVIDIA GPU を搭載した Linux と Windows では、従来の CUDA バックエンドをそのまま利用できます。

### オプション：デスクトップアプリ

<p align="center">
  <img src="./apps/desktop/src-tauri/icons/icon.png" alt="白い正方形に極太サンセリフの m と十分な間隔を空けた上付きの 3 を視覚的に中央配置した mumax3 ultrafast デスクトップアプリのアイコン。" width="112">
</p>

上記の `mumax3-ultrafast` エンジンとコマンドライン操作が基本のインストールであり、macOS デスクトップアプリはその上に任意で追加するインターフェースです。`.mx3` ファイルを正本としたまま、エディタ、作業フォルダの明示的な選択、ワンクリック実行、リアルタイムの磁化とソルバ指標、実行ログ、統合 3D OVF 結果ビューアを利用できます。Live Magnetization の **Open OVF folder** を使えば、アプリ外で作成した結果も直接レンダリングできます。`.ovf` ファイルが直接入っているフォルダを選択してください。結果ビューアの **Projection** では X・Y・Z のスピン成分を選び、−1 から +1 までを結ぶカラーマップの両端色を自由に指定できます（既定は白→黒）。従来の Direction・Magnitude パレットも利用できます。

Apple Developer Program に加入せず、オプションアプリをこの Mac 上で直接ビルドしてインストールできます。

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)"
```

インストーラは最新のタグ付きリリースを解決し、`~/Library/Caches/mumax3-ultrafast` 配下に専用の Node.js・pnpm・Rust ツールを用意します。その後、この Mac でアプリをビルドしてローカルの ad-hoc 署名を付け、`~/Applications/mumax3 ultrafast.app` に配置し、同じリリースの独立エンジンもインストールします。最初のソースビルドには数分かかる場合がありますが、以後は専用キャッシュを再利用します。Apple Command Line Tools がない場合は、システムのインストール画面を完了してから同じコマンドを再実行してください。更新時も同じコマンドを再実行します。エンジン、ビルドキャッシュ、シミュレーションファイルを残してアプリだけを削除するには、次を実行します。

```bash
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)" -- --uninstall
```

このローカルソースビルドは Apple の公証を受けたアプリではないため、管理対象の Mac では制限される場合があります。[GitHub Release](https://github.com/TaewoooPark/mumax3-ultrafast/releases) に `mumax3-ultrafast-app-darwin-arm64.dmg` が含まれている場合は、そちらがより速い署名・公証済みのインストール経路です。オプションと開発手順は [`apps/desktop/README.md`](apps/desktop/README.md) を参照してください。

<table>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/desktop-workspace-editor.png" alt="シミュレーションワークスペースと mx3 エディタを表示したデスクトップアプリ。" width="100%"><br><sub><b>スクリプト作業画面</b> — 選択したフォルダの <code>.mx3</code> ファイルを開くか作成します。</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/desktop-simulation-running.png" alt="ライブ指標とともに実行中のシミュレーションを表示したデスクトップアプリ。" width="100%"><br><sub><b>リアルタイム実行</b> — 磁化、ソルバ指標、進行状況、ログを確認します。</sub></td>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/desktop-results-ready.png" alt="完了したシミュレーションの OVF フレームを表示したデスクトップアプリ。" width="100%"><br><sub><b>結果の準備完了</b> — ワークスペースを離れずに OVF フレームを開きます。</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-overview.png" alt="OVF ベクトル場の対話的な 3D 全体表示。" width="100%"><br><sub><b>3D ベクトル場</b> — 結果を回転、移動、ズーム、色変更、再生します。</sub></td>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-close-up.png" alt="3D 結果ビューアに表示された色付きベクトル矢印の拡大画面。" width="100%"><br><sub><b>詳細表示</b> — グリフを切り替え、X/Y/Z スピン投影を任意の 2 色へマッピングします。</sub></td>
    <td width="50%" valign="top"><img src="./docs/app/result-viewer-top-view.png" alt="デスクトップ結果ビューアに表示された OVF ベクトル場の上面表示。" width="100%"><br><sub><b>上面表示</b> — シミュレーション平面のテクスチャを確認します。</sub></td>
  </tr>
</table>

## シミュレーションの実行

```bash
mumax3 example.mx3            # http://127.0.0.1:35367 のライブ Web UI
mumax3 -http="" example.mx3   # ヘッドレス、ベンチマークやバッチ処理向け
```

出力は `example.out/` に生成されます。どちらのモードも同じシミュレーションを実行し、同じ出力を生成します。Xcode 本体やオフラインの Metal コンパイラは不要で、シェーダライブラリはシステムの Metal ランタイムを通じてコンパイルされます。

---

## ネイティブ移植の構造

| 本家の CUDA コンポーネント | macOS 実装 | 維持された互換性 |
| --- | --- | --- |
| CUDA コンピュートカーネル | Metal コンピュートシェーダと Objective-C++ ランタイムブリッジ | 既存の mumax³ カーネル契約と FP32 モデル |
| cuFFT | MPSGraph FFT、条件を満たす 2 次元変換には同梱の VkFFT 経路 | 反磁場畳み込みが用いるパック済み R2C/C2R レイアウト |
| `cudaMalloc` と CUDA ストリーム | Apple のユニファイドメモリバッファ、コマンドのバッチ化、順序が保証された単一 Metal キュー | バッファの意味論と実行順序 |
| cuRAND の熱雑音 | Philox4x32-10 と Box–Muller による正規分布生成 | Metal でのシード再現性と検証済みの雑音統計 |
| CUDA 専用のプラットフォームバインディング | `darwin/arm64` ビルドタグと互換シム | `.mx3` 言語と上位の Go API |

実装では Apple GPU に適した X 方向連続・32 幅の SIMD タイルを用い、ブロッキングを伴うドレインをまたいで GPU を常駐させ、MPSGraph のスケジューリングが崩れる領域を避けるため大きな反磁場変換を軸ごとに分割し、シェーダを実行時にコンパイルすることで Command Line Tools だけで完結するようにしています。

---

## チューニング

### レイテンシ律速の実行

研究で使うメッシュサイズでは、Apple GPU は演算能力が足りないというより、ホストを待って遊んでいることが多くあります。128×128 では、ホストが Dormand-Prince ステップをエンコードする時間のほうが、GPU がそれを実行する時間より長くなります。これに対処するスイッチが 3 つあり、いずれも既定ビルドの挙動を変えません。

```go
SpeculativeStep = true   // ホストのエンコードと GPU 実行を重ねる。実測 1.53 倍
MinimizeOnGPU   = true   // minimize() の BB ステップ幅をデバイス上に保つ。実測 1.58 倍
```

```sh
mumax3 -j 3 sweep_*.mx3   # GPU あたり N 個の入力をキューイング。minimize() 3 本で合計 2.53 倍
```

`SpeculativeStep` は `MaxErr` を引き続き強制しますが、棄却が 1 ステップ遅れて発生します。そのため時間ステップの列が厳密な制御器とは異なり、数千ステップ後には軌跡が約 1% のレベルで分岐します。`FixDt`、有限の `Temp`、`relax()`、`DemagExtrapolation`、ステップ後フックのもとでは自動的に無効化されます。`MinimizeOnGPU` はすべての降下をビット単位で同一に保ち、収束判定だけが 1 反復遅れるため、最小化が 1 反復先で停止することがあります。いずれもご自身の問題について既定実行と比較検証したうえでお使いください。

### Metal FFT バックエンド

既定では、パディング後の寸法が 512×512 以下の 2 のべき乗で、かつ有効データが厳密な接頭となる 2 次元反磁場変換に対して同梱の VkFFT バックエンドを自動的に用い、それ以外は MPSGraph が処理します。`MUMAX3_METAL_FFT_BACKEND=mps` はすべてのプランを MPSGraph に固定し、`vkfft` はパディング後 1024×1024 の段まで VkFFT を拡張します。

### 任意の反磁場外挿

高次の反磁場外挿は、反磁場の比重が大きいソルバ 4/5/6 の負荷を高速化しますが、近似であるため**既定では無効**です。

```go
SetSolver(5)
DemagExtrapolation = true
```

未対応のソルバや安全でないモデル状態では、厳密な畳み込みへ安全側に倒れます。誤差は軌跡と時間刻みに依存するため、別の問題でのベンチマーク成功は精度の保証になりません。まず[検証結果](bench/demag-extrap/RESULTS.md)と [A/B の手順](bench/demag-extrap/README.md)をご確認ください。

### サイズの選び方

セルあたりのスループットは問題サイズに対して平坦ではありません。測定した M4 では 256² でピークに達し、128² 以下では約 172〜187 µs の固定的な計算あたりオーバーヘッドが支配的になるため、GPU が広くなってもまったく助けになりません。実測曲線は `bench/curve.txt` にあります。

---

## ベンチマークの再現

以上のすべては本リポジトリから再現できます。モデル化した値は一つもありません。

```bash
./bench/capacity.sh name=/path/to/binary   # Metal の制限、セルあたりバイト数、vmmap の内訳、上限
python3 bench/crosstool_svg.py             # 実測データから README の図を再生成
```

シミュレータごとに 1 つずつ、いずれも同じ問題を同じ積分器で解くクロスツールのハーネスは [`crosstool/`](crosstool/) にあります。`run_crosstool.py` が速度と物理、`run_energy.py` が powermetrics、`run_capacity.py` が容量上限を担当します。magnum.np を Apple GPU に載せるために必要なパッチと、MicroMagnetic.jl の Metal バックエンドが実行不可であると結論づけるまでに試したことは、各スクリプトの docstring に記してあります。

結果を実際に左右した測定上の注意を 2 点残しておきます。スワップは `Pageouts` ではなく `vm_stat` の **`Swapouts`** で監視する必要があります。Apple Silicon のメモリコンプレッサはギガバイト単位を書き出しながら `Pageouts` をほとんど動かしません。また、mumax³ の形状ごとの反磁場カーネルキャッシュはこの規模では 1 エントリが 8〜10 GB に達するため、実行のあいだに削除する必要があります。

---

## 主張の範囲

短所を隠すベンチマークには価値がないため、そのまま記します。

- **絶対速度では NVIDIA に負けます。** mumax³ 自身の 62 GPU ベンチマーク表において、M4 は GTX 1650 mobile と GTX 970 のあいだに位置します。主張を Apple Silicon に限定したのは意図的です。すべての軸を 1 台の機械で、TDP の仮定なしに、機械間の正規化なしに測定できる範囲がそこだからです。
- **容量は OOMMF と同率です。** 両者とも 8390 万セルに到達し、両者とも 1 億 490 万で失敗します。mumax3-ultrafast はそのサイズで 4.5 倍速いだけであって、より大きいわけではありません。
- **直前リリースに対する差はメッシュが大きくなるほど縮まります。** 128² で 5.4 倍、2048² で 1.38 倍です。今回の作業が狙ったのは、研究用メッシュが実際に位置するレイテンシ律速の領域でした。

---

## 本家とライセンス

本プロジェクトは [mumax³](https://github.com/mumax/3) から派生しており、その設計と検証は[原論文](https://doi.org/10.1063/1.4899186)に記述されています。NVIDIA CUDA を利用する方は[公式 mumax³ インストール文書](https://mumax.github.io/download.html)に従ってください。

[GNU GPL v3 以降](LICENSE)のもとで配布します。[プロジェクト告知](NOTICE)は本家の CUDA リンク許諾を維持し、macOS 側の変更点を明示しています。Random123、Go、SVGo、Freetype-Go のライセンスと帰属表示は[サードパーティ告知](THIRD_PARTY_NOTICES.md)にまとめてあります。

比較対象は測定時にインストールされていたバージョンで記載しています。[MicroMagnetic.jl](https://github.com/ww1g11/MicroMagnetic.jl) 0.5.0 と Metal.jl 1.10（[arXiv:2406.16064](https://arxiv.org/abs/2406.16064)）、[magnum.np](https://gitlab.com/magnum.np/magnum.np) 2.2.0、[OOMMF](https://math.nist.gov/oommf/) 2.0b0 です。

---

## 作者

私は**パク・テウ（Taewoo Park）**で、韓国科学技術院（KAIST）物理学科の学部生です。2025 年 10 月より、**キム・カプジン教授**が率いる [KAIST 超高速スピンダイナミクス研究室（USDL）](https://spintronics.kaist.ac.kr/)にて、磁気ドメインウォール運動とニューロモルフィックコンピューティング応用に関する実験スピントロニクス研究を行っています。2023 年 6 月から 2024 年 3 月までは、**キム・セグォン教授**の KAIST 量子スピンダイナミクス研究室で、理論モデリングとマイクロマグネティクスシミュレーションによりドメインウォール運動を研究しました。

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>同じ方程式、同じスクリプト、同じ答え。Apple Silicon の上で。</sub></p>
