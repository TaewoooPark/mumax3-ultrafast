#!/bin/bash

# Build and install the optional mumax3-ultrafast desktop app locally.
# Compatible with the Bash 3.2 shipped by macOS.

set -Eeuo pipefail

PROGRAM_NAME="mumax3-ultrafast desktop app source installer"
REPOSITORY_SLUG="${MUMAX3_APP_REPOSITORY_SLUG:-TaewoooPark/mumax3-ultrafast}"
REPOSITORY_URL="${MUMAX3_APP_REPOSITORY_URL:-https://github.com/${REPOSITORY_SLUG}.git}"
MINIMUM_MACOS_MAJOR=14
MINIMUM_NODE_VERSION="22.12.0"
MINIMUM_RUST_VERSION="1.85.0"
PNPM_VERSION="11.9.0"
APP_NAME="mumax3 ultrafast.app"
BUNDLE_IDENTIFIER="io.github.taewoopark.mumax3-ultrafast"
USER_HOME_DIR="${HOME:?HOME is not set}"
INSTALL_DIR="${MUMAX3_APP_INSTALL_DIR:-${USER_HOME_DIR}/Applications}"
CACHE_DIR="${MUMAX3_APP_CACHE_DIR:-${USER_HOME_DIR}/Library/Caches/mumax3-ultrafast/source-build}"
SOURCE_DIR="${MUMAX3_APP_SOURCE_DIR:-}"
VERSION="${MUMAX3_APP_VERSION:-latest}"
SKIP_ENGINE=0
UNINSTALL=0
OPEN_APP=1
CURRENT_STEP="initialization"
TEMP_DIR=""
STAGE_DIR=""
SOURCE_ROOT=""
APP_PATH=""
BACKUP_APP=""
NEW_APP_INSTALLED=0
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export NO_UPDATE_NOTIFIER=1
export PNPM_DISABLE_SELF_UPDATE_CHECK=1
export npm_config_update_notifier=false

log() {
	printf '\n==> %s\n' "$1"
}

note() {
	printf '    %s\n' "$1"
}

warn() {
	printf 'WARNING: %s\n' "$1" >&2
}

fail() {
	printf 'ERROR: %s\n' "$1" >&2
	exit 1
}

cleanup() {
	if [[ -n "$BACKUP_APP" && -d "$BACKUP_APP" && -n "$APP_PATH" ]]; then
		if ((NEW_APP_INSTALLED == 1)) && [[ -d "$APP_PATH" ]]; then
			/bin/rm -rf "$APP_PATH"
		fi
		if [[ ! -e "$APP_PATH" ]]; then
			/bin/mv "$BACKUP_APP" "$APP_PATH" || true
		fi
	fi
	if [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]]; then
		/bin/rm -rf "$STAGE_DIR"
	fi
	if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
		/bin/rm -rf "$TEMP_DIR"
	fi
}

on_error() {
	status=$?
	trap - ERR
	printf '\nERROR: %s failed during: %s\n' "$PROGRAM_NAME" "$CURRENT_STEP" >&2
	printf '       Exit status: %s\n' "$status" >&2
	printf '       Fix the reported error and run the installer again.\n' >&2
	exit "$status"
}

on_interrupt() {
	trap - INT TERM
	printf '\nERROR: Installation interrupted during: %s\n' "$CURRENT_STEP" >&2
	exit 130
}

trap cleanup EXIT
trap on_error ERR
trap on_interrupt INT TERM

usage() {
	cat <<'EOF'
Usage: install-app-macos.sh [options]

Build the optional mumax3-ultrafast desktop app from a tagged source release,
apply a local ad-hoc signature, and install it in ~/Applications. The standalone
engine remains the primary installation and is installed separately by the
verified engine installer. Re-run the same command to update the app.

Options:
  --version VERSION  Build a release tag. Default: latest GitHub release.
  --install-dir DIR  Install the app into DIR. Default: ~/Applications.
  --source-dir DIR   Build an existing checkout instead of cloning a tag.
  --skip-engine      Do not install or update the standalone engine.
  --no-open          Do not launch the app after installation.
  --uninstall        Remove the locally installed app only.
  -h, --help         Show this help.

Environment overrides:
  MUMAX3_APP_VERSION          Same as --version.
  MUMAX3_APP_INSTALL_DIR      Same as --install-dir.
  MUMAX3_APP_SOURCE_DIR       Same as --source-dir.
  MUMAX3_APP_CACHE_DIR        Private Node, pnpm, Rust, and build cache.
  MUMAX3_APP_REPOSITORY_SLUG  GitHub owner/repository used for releases.
  MUMAX3_APP_REPOSITORY_URL   Git repository cloned for source builds.

The final app is built on this Mac and is not Apple-notarized. Managed Macs may
still restrict locally built applications. The first run may ask for normal
folder access when a simulation workspace is selected.
EOF
}

while (($# > 0)); do
	case "$1" in
		--version)
			(($# >= 2)) || fail "--version requires a release tag"
			VERSION=$2
			shift 2
			;;
		--install-dir)
			(($# >= 2)) || fail "--install-dir requires a directory"
			INSTALL_DIR=$2
			shift 2
			;;
		--source-dir)
			(($# >= 2)) || fail "--source-dir requires a directory"
			SOURCE_DIR=$2
			shift 2
			;;
		--skip-engine)
			SKIP_ENGINE=1
			shift
			;;
		--no-open)
			OPEN_APP=0
			shift
			;;
		--uninstall)
			UNINSTALL=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			fail "unknown option: $1"
			;;
	esac
done

version_at_least() {
	/usr/bin/awk -v actual="$1" -v required="$2" '
		BEGIN {
			split(actual, a, ".")
			split(required, r, ".")
			for (i = 1; i <= 3; i++) {
				av = (a[i] == "" ? 0 : a[i]) + 0
				rv = (r[i] == "" ? 0 : r[i]) + 0
				if (av > rv) exit 0
				if (av < rv) exit 1
			}
			exit 0
		}
	'
}

check_mac_compatibility() {
	machine_arch=$(/usr/bin/uname -m)
	if [[ "$machine_arch" != "arm64" ]]; then
		if [[ "$machine_arch" == "x86_64" ]] &&
			[[ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" == "1" ]]; then
			fail "Terminal is running through Rosetta. Open a native Apple Silicon Terminal and retry."
		fi
		fail "unsupported architecture '$machine_arch'; the desktop app requires Apple Silicon"
	fi

	macos_version=$(/usr/bin/sw_vers -productVersion)
	macos_major=${macos_version%%.*}
	case "$macos_major" in
		''|*[!0-9]*) fail "could not parse macOS version '$macos_version'" ;;
	esac
	((macos_major >= MINIMUM_MACOS_MAJOR)) ||
		fail "macOS ${MINIMUM_MACOS_MAJOR} or newer is required; found $macos_version"
	note "Apple Silicon, macOS $macos_version"
}

bundle_identifier() {
	/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$1/Contents/Info.plist" 2>/dev/null
}

safe_app_path() {
	app_path=${INSTALL_DIR%/}/${APP_NAME}
	case "$app_path" in
		"/${APP_NAME}"|"${USER_HOME_DIR}"|"${INSTALL_DIR}")
			fail "refusing unsafe application path '$app_path'"
			;;
	esac
	printf '%s\n' "$app_path"
}

uninstall_app() {
	app_path=$(safe_app_path)
	if [[ ! -e "$app_path" ]]; then
		note "No app is installed at $app_path"
		return
	fi
	[[ -d "$app_path" ]] || fail "refusing to remove non-application path '$app_path'"
	installed_identifier=$(bundle_identifier "$app_path" || true)
	[[ "$installed_identifier" == "$BUNDLE_IDENTIFIER" ]] ||
		fail "refusing to remove an app not owned by this installer: $app_path"
	/bin/rm -rf "$app_path"
	note "Removed $app_path"
	note "The standalone engine, build cache, and simulation files were preserved."
}

ensure_command_line_tools() {
	if /usr/bin/xcrun --find clang >/dev/null 2>&1 &&
		/usr/bin/xcrun --show-sdk-path >/dev/null 2>&1; then
		return
	fi
	warn "Apple Command Line Tools are required for the one-time local build."
	/usr/bin/xcode-select --install >/dev/null 2>&1 || true
	fail "finish the Apple Command Line Tools installation, then run this command again"
}

resolve_version() {
	if [[ "$VERSION" == "latest" ]]; then
		CURRENT_STEP="resolving the latest release tag"
		latest_url=$(
			/usr/bin/curl --fail --location --silent --show-error \
				--output /dev/null --write-out '%{url_effective}' \
				"https://github.com/${REPOSITORY_SLUG}/releases/latest"
		)
		VERSION=${latest_url##*/}
	fi
	case "$VERSION" in
		''|latest|*[!A-Za-z0-9._+-]*) fail "invalid release version '$VERSION'" ;;
	esac
	note "Source release: $VERSION"
}

prepare_source() {
	CURRENT_STEP="preparing the ${VERSION} source"
	if [[ -n "$SOURCE_DIR" ]]; then
		[[ -d "$SOURCE_DIR/apps/desktop" ]] ||
			fail "source directory does not contain apps/desktop: $SOURCE_DIR"
		SOURCE_ROOT=$(cd "$SOURCE_DIR" && /bin/pwd -P)
		return
	fi

	SOURCE_ROOT="$TEMP_DIR/source"
	/usr/bin/git clone --quiet --depth 1 --branch "$VERSION" "$REPOSITORY_URL" "$SOURCE_ROOT"
	head_commit=$(/usr/bin/git -C "$SOURCE_ROOT" rev-parse HEAD)
	expected_commit=$(
		/usr/bin/git ls-remote "$REPOSITORY_URL" "refs/tags/${VERSION}^{}" |
			/usr/bin/awk 'NR == 1 { print $1 }'
	)
	if [[ -z "$expected_commit" ]]; then
		expected_commit=$(
			/usr/bin/git ls-remote "$REPOSITORY_URL" "refs/tags/${VERSION}" |
				/usr/bin/awk 'NR == 1 { print $1 }'
		)
	fi
	[[ -n "$expected_commit" && "$head_commit" == "$expected_commit" ]] ||
		fail "the cloned commit does not match tag $VERSION"
}

ensure_node() {
	CURRENT_STEP="preparing Node.js"
	if command -v node >/dev/null 2>&1; then
		node_version=$(node -p 'process.versions.node' 2>/dev/null || true)
		if [[ -n "$node_version" ]] && version_at_least "$node_version" "$MINIMUM_NODE_VERSION"; then
			note "Using Node.js $node_version"
			return
		fi
	fi

	/bin/mkdir -p "$CACHE_DIR/node"
	checksums="$TEMP_DIR/node-shasums.txt"
	/usr/bin/curl --fail --location --silent --show-error \
		-o "$checksums" "https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt"
	node_line=$(
		/usr/bin/awk '$2 ~ /^node-v[0-9.]+-darwin-arm64[.]tar[.]gz$/ { print; exit }' "$checksums"
	)
	[[ -n "$node_line" ]] || fail "could not find the Apple Silicon Node.js archive"
	node_sha=${node_line%% *}
	node_archive=${node_line##* }
	case "$node_archive" in
		node-v*-darwin-arm64.tar.gz) ;;
		*) fail "unexpected Node.js archive name '$node_archive'" ;;
	esac
	node_directory=${node_archive%-darwin-arm64.tar.gz}
	node_root="$CACHE_DIR/node/$node_directory"
	if [[ ! -x "$node_root/bin/node" ]]; then
		node_download="$TEMP_DIR/$node_archive"
		/usr/bin/curl --fail --location --silent --show-error \
			-o "$node_download" "https://nodejs.org/dist/latest-v24.x/$node_archive"
		printf '%s  %s\n' "$node_sha" "$node_download" |
			/usr/bin/shasum -a 256 -c - >/dev/null
		node_stage="$TEMP_DIR/$node_directory"
		/bin/mkdir -p "$node_stage"
		/usr/bin/tar -xzf "$node_download" -C "$node_stage" --strip-components 1
		/bin/mv "$node_stage" "$node_root"
	fi
	export PATH="$node_root/bin:$PATH"
	node_version=$(node -p 'process.versions.node')
	version_at_least "$node_version" "$MINIMUM_NODE_VERSION" ||
		fail "downloaded Node.js $node_version is too old"
	note "Using private Node.js $node_version"
}

ensure_pnpm() {
	CURRENT_STEP="preparing pnpm"
	pnpm_root="$CACHE_DIR/pnpm/$PNPM_VERSION"
	if [[ ! -x "$pnpm_root/node_modules/.bin/pnpm" ]]; then
		/bin/mkdir -p "$pnpm_root"
		npm install --prefix "$pnpm_root" --ignore-scripts --no-audit --no-fund \
			"pnpm@$PNPM_VERSION" >/dev/null
	fi
	export PATH="$pnpm_root/node_modules/.bin:$PATH"
	[[ "$(pnpm --version)" == "$PNPM_VERSION" ]] || fail "could not prepare pnpm $PNPM_VERSION"
	note "Using private pnpm $PNPM_VERSION"
}

ensure_rust() {
	CURRENT_STEP="preparing Rust"
	if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
		rust_version=$(rustc --version 2>/dev/null | /usr/bin/awk '{ print $2 }' || true)
		if version_at_least "$rust_version" "$MINIMUM_RUST_VERSION"; then
			note "Using Rust $rust_version"
			return
		fi
	fi

	export CARGO_HOME="$CACHE_DIR/cargo-home"
	export RUSTUP_HOME="$CACHE_DIR/rustup-home"
	export PATH="$CARGO_HOME/bin:$PATH"
	if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
		/bin/mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
		rustup_init="$TEMP_DIR/rustup-init"
		rustup_sha="$TEMP_DIR/rustup-init.sha256"
		rustup_url="https://static.rust-lang.org/rustup/dist/aarch64-apple-darwin/rustup-init"
		/usr/bin/curl --fail --location --silent --show-error -o "$rustup_init" "$rustup_url"
		/usr/bin/curl --fail --location --silent --show-error -o "$rustup_sha" "${rustup_url}.sha256"
		expected_rustup_sha=$(/usr/bin/awk 'NR == 1 { print $1 }' "$rustup_sha")
		printf '%s  %s\n' "$expected_rustup_sha" "$rustup_init" |
			/usr/bin/shasum -a 256 -c - >/dev/null
		/bin/chmod +x "$rustup_init"
		"$rustup_init" -y --no-modify-path --profile minimal --default-toolchain stable >/dev/null
	fi
	if ! rustc --version >/dev/null 2>&1; then
		"$CARGO_HOME/bin/rustup" default stable >/dev/null
	fi
	rust_version=$(rustc --version | /usr/bin/awk '{ print $2 }')
	version_at_least "$rust_version" "$MINIMUM_RUST_VERSION" ||
		fail "prepared Rust $rust_version is too old"
	note "Using private Rust $rust_version"
}

build_app() {
	CURRENT_STEP="building the desktop app locally"
	log "Building mumax3 ultrafast.app"
	build_log="$TEMP_DIR/desktop-build.log"
	export CARGO_TARGET_DIR="$CACHE_DIR/cargo-target"
	export MACOSX_DEPLOYMENT_TARGET="14.0"
	export npm_config_cache="$CACHE_DIR/npm-cache"
	/bin/mkdir -p "$CACHE_DIR/pnpm-store" "$CARGO_TARGET_DIR"
	if ! (
		cd "$SOURCE_ROOT/apps/desktop"
		pnpm install --frozen-lockfile --store-dir "$CACHE_DIR/pnpm-store"
		pnpm tauri build --bundles app
	) >"$build_log" 2>&1; then
		warn "the local build failed; the final build output follows"
		/usr/bin/tail -n 120 "$build_log" >&2
		fail "could not build the desktop app"
	fi
	note "Local application build completed"
	BUILT_APP=$(
		/usr/bin/find "$CARGO_TARGET_DIR/release/bundle/macos" -maxdepth 1 \
			-name '*.app' -type d -print -quit
	)
	[[ -n "$BUILT_APP" ]] || fail "the Tauri build did not produce a macOS app bundle"
	[[ "$(bundle_identifier "$BUILT_APP")" == "$BUNDLE_IDENTIFIER" ]] ||
		fail "the built app has an unexpected bundle identifier"
}

install_engine() {
	if ((SKIP_ENGINE == 1)); then
		note "Standalone engine installation skipped"
		return
	fi
	CURRENT_STEP="installing the standalone engine"
	log "Installing the matching standalone engine"
	/bin/bash "$SOURCE_ROOT/install-macos.sh" --version "$VERSION"
}

install_app() {
	CURRENT_STEP="installing the locally built application"
	log "Installing the desktop app"
	APP_PATH=$(safe_app_path)
	/bin/mkdir -p "$INSTALL_DIR"
	STAGE_DIR=$(/usr/bin/mktemp -d "${INSTALL_DIR%/}/.mumax3-ultrafast-install.XXXXXX")
	staged_app="$STAGE_DIR/$APP_NAME"
	/usr/bin/ditto "$BUILT_APP" "$staged_app"
	/bin/mkdir -p "$staged_app/Contents/Resources"
	printf '%s\n' "$VERSION" >"$staged_app/Contents/Resources/source-install-version"
	/usr/bin/codesign --force --deep --sign - --timestamp=none "$staged_app"
	/usr/bin/codesign --verify --deep --strict "$staged_app"
	[[ "$(bundle_identifier "$staged_app")" == "$BUNDLE_IDENTIFIER" ]] ||
		fail "the staged app has an unexpected bundle identifier"

	BACKUP_APP="${INSTALL_DIR%/}/.mumax3-ultrafast-backup.$$"
	[[ ! -e "$BACKUP_APP" ]] || fail "temporary backup path already exists: $BACKUP_APP"
	if [[ -e "$APP_PATH" ]]; then
		[[ -d "$APP_PATH" ]] || fail "refusing to replace non-application path '$APP_PATH'"
		installed_identifier=$(bundle_identifier "$APP_PATH" || true)
		[[ "$installed_identifier" == "$BUNDLE_IDENTIFIER" ]] ||
			fail "refusing to replace an app not owned by this installer: $APP_PATH"
		/bin/mv "$APP_PATH" "$BACKUP_APP"
	fi
	if ! /bin/mv "$staged_app" "$APP_PATH"; then
		if [[ -d "$BACKUP_APP" ]]; then
			/bin/mv "$BACKUP_APP" "$APP_PATH"
		fi
		BACKUP_APP=""
		fail "could not move the app into $INSTALL_DIR"
	fi
	NEW_APP_INSTALLED=1
	if [[ -d "$BACKUP_APP" ]]; then
		/bin/rm -rf "$BACKUP_APP"
	fi
	BACKUP_APP=""
	NEW_APP_INSTALLED=0
	note "Installed: $APP_PATH"
	note "Build cache: $CACHE_DIR"
	note "Re-run this installer to build and install a newer release."
	if ((OPEN_APP == 1)); then
		/usr/bin/open "$APP_PATH" || warn "the app was installed but could not be opened automatically"
	fi
}

if ((UNINSTALL == 1)); then
	log "Removing the locally built desktop app"
	uninstall_app
	exit 0
fi

log "Checking this Mac"
check_mac_compatibility
ensure_command_line_tools
resolve_version
TEMP_DIR=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/mumax3-ultrafast-app.XXXXXX")
prepare_source
ensure_node
ensure_pnpm
ensure_rust
build_app
install_engine
install_app

log "Local source installation complete"
note "The app stays in $(safe_app_path) after the temporary source checkout is removed."
