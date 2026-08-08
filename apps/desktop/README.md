# mumax³ ultrafast desktop

<p align="center">
  <img src="./src-tauri/icons/icon.png" alt="mumax3 ultrafast desktop app icon with a clearly separated superscript 3" width="112">
</p>

This directory contains the initial lightweight desktop workspace for
`mumax3-ultrafast`. It deliberately keeps the existing mumax³ engine, `.mx3`
language, output directories, and HTTP viewer intact. The wrapper only manages
the user-facing workflow around them.

## Current workflow

1. Choose a working folder.
2. Open or edit an `.mx3` script.
3. Save the script explicitly or let **Run simulation** save it before launch.
4. Install the standalone `mumax3-ultrafast` engine if it is not already on the
   Mac. The desktop app is an optional client and does not bundle or replace it.
5. Select **Run simulation** to start a private loopback viewer on a free port.
6. Watch the live magnetization, solver values, and process output.
7. After the script completes, select **View result** to open the integrated 3D
   OVF viewer. Rotate, pan, zoom, recolor, switch glyph styles, and play saved
   field frames without keeping the simulation process alive. Projection colors
   map a selected X, Y, or Z spin component from −1 to +1 between two editable
   endpoints, initially white to black.
8. To inspect results generated elsewhere, select **Open OVF folder** in Live
   Magnetization and choose the folder that directly contains the `.ovf` files.
   The result workspace opens automatically; an empty selection prompts for the
   correct folder instead.

Every run receives a unique timestamped `.out` directory. Existing result
directories are never cleaned or overwritten by the desktop application.

## Local source installation

The engine remains the default standalone product. To add the optional app
without an Apple Developer Program membership, run this command on Apple
Silicon with macOS 14 or newer:

```sh
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)"
```

The installer checks out the latest tagged release, keeps its Node.js, pnpm,
Rust, dependency, and compiler caches under
`~/Library/Caches/mumax3-ultrafast/source-build`, builds and ad-hoc signs the
Tauri bundle locally, installs it to `~/Applications/mumax3 ultrafast.app`, and
installs the matching standalone engine through the verified binary-first
engine installer. It does not modify a global Node.js, pnpm, or Rust setup.
The first source build can take several minutes; subsequent runs reuse the
private compiler and dependency caches.

Apple Command Line Tools are the only system build prerequisite. If they are
absent, macOS opens its installer; finish that prompt and run the command again.
Re-run the same command for an update. Remove only the application with:

```sh
/bin/bash -c "$(/usr/bin/curl -fsSL https://raw.githubusercontent.com/TaewoooPark/mumax3-ultrafast/main/install-app-macos.sh)" -- --uninstall
```

Useful direct options are `--version TAG`, `--install-dir DIR`, `--source-dir
DIR`, `--skip-engine`, and `--no-open`. The local build is intentionally
identified as ad-hoc signed rather than Apple-notarized; managed Macs can still
reject it. A signed and notarized DMG, when attached to a GitHub Release,
remains the faster distribution path.

## Development

Requirements:

- Apple Silicon on macOS 14 or newer
- Go 1.22.4 or newer
- Rust and Cargo
- Node.js and pnpm

From this directory:

```sh
pnpm install
pnpm desktop
```

The development build discovers a compatible standalone engine in this order:

1. `MUMAX3_ULTRAFAST_BIN` or `MUMAX3_BINARY`
2. `.mumax3-ultrafast/bin/mumax3` in a developer checkout
3. `~/.local/bin/mumax3`
4. `/opt/homebrew/bin/mumax3`
5. `/usr/local/bin/mumax3`
6. the inherited `PATH`

Every candidate must return the `mumax3-ultrafast` Metal desktop compatibility
identifier before the app will run it, preventing an older upstream or
`mumax3-for-mac` executable from being selected accidentally. Distributed app
builds do not search for a source checkout. Set `MUMAX3_ULTRAFAST_HOME` only to
enable the developer-only **Build dev engine** fallback; its output is written
to `.mumax3-ultrafast/bin/mumax3` inside that checkout.

The standalone engine remains installed if the optional desktop app is removed.
The engine installer replaces an existing `mumax3` only after checksum,
compatibility, and Metal smoke tests pass. Run `install-macos.sh --uninstall`
with the same `--install-dir` to remove the engine without deleting simulation
files or other commands in that directory.

## Architecture

- React and TypeScript provide the two-pane workspace and lightweight editor.
- Tauri and Rust own file access, builds, child-process lifetime, and local HTTP
  polling.
- The existing mumax³ `/render/m` endpoint supplies live JPEG frames.
- The existing GUI update endpoint supplies step, time, timestep, torque,
  integration error, and progress values.
- A streaming Rust parser reads OVF Text, Binary 4, and Binary 8 results and
  samples large meshes to a bounded glyph count.
- Three.js renders result frames as interactive arrows or cuboids with
  customizable X/Y/Z projection maps plus direction and magnitude color modes.

The visual tokens are adapted from Harness Router under Apache-2.0. Attribution
is recorded in [`NOTICE`](NOTICE).

## Local security model

- The simulation viewer binds to `127.0.0.1` on a dynamically selected port.
- GUI metric requests reject non-loopback URLs.
- Script names cannot contain parent or nested directory components.
- The wrapper writes only to the working folder selected by the user.
- Result loading is confined to OVF files inside the active run directory or a
  directory explicitly selected through the native folder picker.
- Engine stdin is closed and all stdout and stderr are captured for display.
- Completed simulations exit normally; closing the application terminates an
  active managed simulation process.

## Validation

```sh
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo test --manifest-path src-tauri/Cargo.toml
go test -vet=off ./cmd/mumax3 ./engine
```

The repository's historical Go vet warnings are independent of the desktop
wrapper, so the scoped engine tests disable vet while still executing the test
suites.
