# mumax³ ultrafast desktop

This directory contains the initial lightweight desktop workspace for
`mumax3-ultrafast`. It deliberately keeps the existing mumax³ engine, `.mx3`
language, output directories, and HTTP viewer intact. The wrapper only manages
the user-facing workflow around them.

## Current workflow

1. Choose a working folder.
2. Open or edit an `.mx3` script.
3. Save the script explicitly or let **Run simulation** save it before launch.
4. Select **Build engine** to compile the native Go and Metal executable.
5. Select **Run simulation** to start a private loopback viewer on a free port.
6. Watch the live magnetization, solver values, and process output.
7. After the script completes, select **View result** to open the integrated 3D
   OVF viewer. Rotate, pan, zoom, recolor, switch glyph styles, and play saved
   field frames without keeping the simulation process alive.

Every run receives a unique timestamped `.out` directory. Existing result
directories are never cleaned or overwritten by the desktop application.

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

The application currently expects to run from a source checkout. Repository
discovery follows this order:

1. `MUMAX3_ULTRAFAST_HOME`
2. the current directory and its parents
3. the repository containing the compiled Tauri crate

The managed engine is written to
`.mumax3-ultrafast/bin/mumax3-ultrafast` at the repository root.

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
  direction and magnitude color modes.

The visual tokens are adapted from Harness Router under Apache-2.0. Attribution
is recorded in [`NOTICE`](NOTICE).

## Local security model

- The simulation viewer binds to `127.0.0.1` on a dynamically selected port.
- GUI metric requests reject non-loopback URLs.
- Script names cannot contain parent or nested directory components.
- The wrapper writes only to the working folder selected by the user.
- Result loading is confined to OVF files inside the current run directory.
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
