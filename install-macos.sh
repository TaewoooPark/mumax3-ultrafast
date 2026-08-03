#!/bin/bash

# Install the standalone mumax3-ultrafast engine on Apple Silicon.
# Compatible with the Bash 3.2 shipped by macOS.

set -Eeuo pipefail

PROGRAM_NAME="mumax3-ultrafast macOS installer"
REPOSITORY_SLUG="${MUMAX3_REPOSITORY_SLUG:-TaewoooPark/mumax3-ultrafast}"
REPOSITORY_URL="${MUMAX3_REPOSITORY_URL:-https://github.com/${REPOSITORY_SLUG}.git}"
RELEASE_BASE_URL="${MUMAX3_RELEASE_BASE_URL:-}"
MINIMUM_MACOS_MAJOR=14
MINIMUM_GO_VERSION="1.22.4"
ARCHIVE_NAME="mumax3-ultrafast-darwin-arm64.tar.gz"
CHECKSUM_NAME="${ARCHIVE_NAME}.sha256"
CURRENT_STEP="initialization"
INSTALL_DIR="${MUMAX3_INSTALL_DIR:-${HOME:?HOME is not set}/.local/bin}"
SOURCE_DIR="${MUMAX3_SOURCE_DIR:-}"
VERSION="${MUMAX3_VERSION:-latest}"
FROM_SOURCE=0
NO_PROFILE=0
UNINSTALL=0
USER_HOME_DIR="${HOME:?HOME is not set}"
TEMP_DIR=""
STAGED_BINARY=""
PROBE_FILE=""
PROBE_PID=""
PROBE_RESULT=""
export PATH="/opt/homebrew/bin:/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

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
	if [[ -n "$PROBE_PID" ]]; then
		/bin/kill -KILL "$PROBE_PID" >/dev/null 2>&1 || true
	fi
	if [[ -n "$PROBE_FILE" && -f "$PROBE_FILE" ]]; then
		/bin/rm -f "$PROBE_FILE"
	fi
	if [[ -n "$STAGED_BINARY" && -f "$STAGED_BINARY" ]]; then
		/bin/rm -f "$STAGED_BINARY"
	fi
	if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
		/bin/rm -rf "$TEMP_DIR"
	fi
}

bounded_engine_probe() {
	executable=$1
	PROBE_RESULT=""
	PROBE_FILE=$(/usr/bin/mktemp "${TMPDIR:-/tmp}/mumax3-probe.XXXXXX")
	"$executable" -ultrafast-probe >"$PROBE_FILE" 2>/dev/null &
	PROBE_PID=$!
	probe_ticks=0
	while /bin/kill -0 "$PROBE_PID" >/dev/null 2>&1; do
		if ((probe_ticks >= 40)); then
			/bin/kill -KILL "$PROBE_PID" >/dev/null 2>&1 || true
			PROBE_PID=""
			/bin/rm -f "$PROBE_FILE"
			PROBE_FILE=""
			return 124
		fi
		/bin/sleep 0.05
		probe_ticks=$((probe_ticks + 1))
	done
	if wait "$PROBE_PID"; then
		probe_status=0
	else
		probe_status=$?
	fi
	PROBE_PID=""
	probe_output=$(<"$PROBE_FILE")
	/bin/rm -f "$PROBE_FILE"
	PROBE_FILE=""
	((probe_status == 0)) || return "$probe_status"
	PROBE_RESULT=$probe_output
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
Usage: install-macos.sh [options]

Install the standalone mumax3-ultrafast engine on Apple Silicon. The default
path uses a prebuilt release when available and does not install the optional
desktop app. Until a release artifact exists, it falls back to a source build
and may install Apple Command Line Tools, Homebrew, and Go.

Options:
  --version VERSION  Install a release tag such as v0.2.0. Default: latest.
  --install-dir DIR  Install the mumax3 command into DIR. Default: ~/.local/bin.
  --no-profile       Do not add the installation directory to ~/.zprofile.
  --from-source      Build from source instead of downloading a release.
  --source-dir DIR   With --from-source, clone into or build from DIR.
                      Default: the current checkout or ~/mumax3-ultrafast.
  --uninstall        Remove mumax3 from the selected installation directory.
  -h, --help         Show this help.

Environment overrides:
  MUMAX3_VERSION             Same as --version.
  MUMAX3_INSTALL_DIR         Same as --install-dir.
  MUMAX3_SOURCE_DIR          Same as --source-dir.
  MUMAX3_REPOSITORY_SLUG     GitHub owner/repository used for releases.
  MUMAX3_REPOSITORY_URL      Repository cloned by --from-source.
  MUMAX3_RELEASE_BASE_URL    Release download base URL for mirrors and tests.
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
		--no-profile)
			NO_PROFILE=1
			shift
			;;
		--from-source)
			FROM_SOURCE=1
			shift
			;;
		--source-dir)
			(($# >= 2)) || fail "--source-dir requires a directory"
			SOURCE_DIR=$2
			FROM_SOURCE=1
			shift 2
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
		fail "unsupported architecture '$machine_arch'; the Metal backend requires Apple Silicon"
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

resolve_profile_path() {
	if [[ -n "${ZDOTDIR:-}" ]]; then
		case "$ZDOTDIR" in
			/*) printf '%s/.zprofile\n' "${ZDOTDIR%/}" ;;
			*) printf '%s/%s/.zprofile\n' "${USER_HOME_DIR%/}" "${ZDOTDIR%/}" ;;
		esac
	else
		printf '%s/.zprofile\n' "${USER_HOME_DIR%/}"
	fi
}

ensure_profile_path() {
	if ((NO_PROFILE == 1)); then
		return
	fi

	profile_path=$(resolve_profile_path)
	profile_dir=$(/usr/bin/dirname "$profile_path")
	/bin/mkdir -p "$profile_dir"
	[[ -e "$profile_path" ]] || /usr/bin/touch "$profile_path"
	printf -v quoted_install_dir '%q' "$INSTALL_DIR"
	profile_line="export PATH=${quoted_install_dir}:\$PATH"
	if ! /usr/bin/grep -Fqx "$profile_line" "$profile_path"; then
		{
			printf '\n# Added by the mumax3-ultrafast installer\n'
			printf '%s\n' "$profile_line"
		} >>"$profile_path"
		note "Updated $profile_path"
	fi
}

release_download_base() {
	if [[ -n "$RELEASE_BASE_URL" ]]; then
		printf '%s\n' "${RELEASE_BASE_URL%/}"
	elif [[ "$VERSION" == "latest" ]]; then
		printf 'https://github.com/%s/releases/latest/download\n' "$REPOSITORY_SLUG"
	else
		printf 'https://github.com/%s/releases/download/%s\n' "$REPOSITORY_SLUG" "$VERSION"
	fi
}

release_available() {
	base_url=$(release_download_base)
	http_code=$(
		/usr/bin/curl --fail --location --head --silent --show-error \
			--output /dev/null --write-out '%{http_code}' \
			"${base_url}/${ARCHIVE_NAME}"
	)
	curl_status=$?
	if ((curl_status == 0)); then
		return 0
	fi
	if [[ "$http_code" == "404" ]]; then
		return 1
	fi
	return 2
}

download_release() {
	CURRENT_STEP="downloading the ${VERSION} engine release"
	log "Downloading mumax3-ultrafast ${VERSION}"
	TEMP_DIR=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/mumax3-ultrafast.XXXXXX")
	base_url=$(release_download_base)
	/usr/bin/curl --fail --location --retry 3 --silent --show-error \
		"${base_url}/${ARCHIVE_NAME}" -o "${TEMP_DIR}/${ARCHIVE_NAME}"
	/usr/bin/curl --fail --location --retry 3 --silent --show-error \
		"${base_url}/${CHECKSUM_NAME}" -o "${TEMP_DIR}/${CHECKSUM_NAME}"

	CURRENT_STEP="verifying the release checksum"
	(
		cd "$TEMP_DIR"
		/usr/bin/shasum -a 256 -c "$CHECKSUM_NAME"
	)

	CURRENT_STEP="extracting the engine release"
	/usr/bin/tar -xzf "${TEMP_DIR}/${ARCHIVE_NAME}" -C "$TEMP_DIR"
	[[ -f "${TEMP_DIR}/mumax3" ]] ||
		fail "the release archive does not contain the mumax3 executable"

	CURRENT_STEP="staging the engine executable"
	/bin/mkdir -p "$INSTALL_DIR"
	STAGED_BINARY=$(/usr/bin/mktemp "${INSTALL_DIR%/}/.mumax3.new.XXXXXX")
	/usr/bin/install -m 0755 "${TEMP_DIR}/mumax3" "$STAGED_BINARY"
}

developer_tools_ready() {
	/usr/bin/xcrun --find clang >/dev/null 2>&1 &&
		command -v git >/dev/null 2>&1
}

ensure_developer_tools() {
	if developer_tools_ready; then
		note "Apple Command Line Tools are ready."
		return
	fi
	if /usr/bin/xcode-select -p >/dev/null 2>&1; then
		fail "developer tools are selected but clang or git is unavailable. Repair the active Xcode installation."
	fi

	CURRENT_STEP="starting the Apple Command Line Tools installer"
	log "Apple Command Line Tools are required for --from-source"
	/usr/bin/xcode-select --install
	CURRENT_STEP="waiting for Apple Command Line Tools"
	elapsed=0
	while ! developer_tools_ready; do
		if ((elapsed >= 7200)); then
			fail "timed out waiting for Command Line Tools after 120 minutes"
		fi
		/bin/sleep 5
		elapsed=$((elapsed + 5))
	done
}

ensure_native_homebrew() {
	if [[ ! -x /opt/homebrew/bin/brew ]]; then
		CURRENT_STEP="installing native Apple Silicon Homebrew for the source fallback"
		log "Installing the source-build toolchain"
		homebrew_installer=$(
			/usr/bin/curl -fsSL \
				https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
		)
		/bin/bash -c "$homebrew_installer"
	fi

	[[ -x /opt/homebrew/bin/brew ]] ||
		fail "native Homebrew was not installed at /opt/homebrew"
	eval "$(/opt/homebrew/bin/brew shellenv)"
}

go_is_compatible() {
	command -v go >/dev/null 2>&1 || return 1
	go_version_raw=$(go env GOVERSION 2>/dev/null) || return 1
	go_version=${go_version_raw#go}
	version_at_least "$go_version" "$MINIMUM_GO_VERSION" &&
		[[ "$(go env GOHOSTOS 2>/dev/null)" == "darwin" ]] &&
		[[ "$(go env GOHOSTARCH 2>/dev/null)" == "arm64" ]] &&
		[[ "$(go env GOOS 2>/dev/null)" == "darwin" ]] &&
		[[ "$(go env GOARCH 2>/dev/null)" == "arm64" ]]
}

ensure_go() {
	if go_is_compatible; then
		note "Compatible Go found: $(go version)"
		return
	fi

	ensure_native_homebrew
	CURRENT_STEP="installing Go ${MINIMUM_GO_VERSION} or newer"
	if /opt/homebrew/bin/brew list --versions go >/dev/null 2>&1; then
		/opt/homebrew/bin/brew upgrade go
	else
		/opt/homebrew/bin/brew install go
	fi
	hash -r
	go_is_compatible ||
		fail "Go must be ${MINIMUM_GO_VERSION}+ and target native darwin/arm64"
	note "Installed $(go version)"
}

is_metal_source_tree() {
	candidate_dir=$1
	[[ -f "$candidate_dir/go.mod" ]] &&
		/usr/bin/grep -Eq '^module[[:space:]]+github\.com/mumax/3$' "$candidate_dir/go.mod" &&
		[[ -s "$candidate_dir/cuda/metal/kernels/mumax3_kernels.metal" ]]
}

detect_local_checkout() {
	script_path="${BASH_SOURCE[0]:-}"
	[[ -n "$script_path" && -f "$script_path" ]] || return 1
	script_dir=$(CDPATH= cd -- "$(/usr/bin/dirname "$script_path")" && /bin/pwd)
	is_metal_source_tree "$script_dir" || return 1
	printf '%s\n' "$script_dir"
}

prepare_source() {
	if [[ -z "$SOURCE_DIR" ]]; then
		if local_checkout=$(detect_local_checkout); then
			SOURCE_DIR=$local_checkout
		else
			SOURCE_DIR="${USER_HOME_DIR%/}/mumax3-ultrafast"
		fi
	fi
	case "$SOURCE_DIR" in
		/*) ;;
		*) SOURCE_DIR="$PWD/$SOURCE_DIR" ;;
	esac

	if [[ ! -e "$SOURCE_DIR" ]]; then
		CURRENT_STEP="cloning mumax3-ultrafast"
		log "Cloning mumax3-ultrafast"
		if [[ "$VERSION" == "latest" ]]; then
			/usr/bin/git clone --depth 1 "$REPOSITORY_URL" "$SOURCE_DIR"
		else
			/usr/bin/git clone --depth 1 --branch "$VERSION" "$REPOSITORY_URL" "$SOURCE_DIR"
		fi
	elif is_metal_source_tree "$SOURCE_DIR"; then
		note "Using existing source directory: $SOURCE_DIR"
	else
		fail "$SOURCE_DIR already exists and is not a mumax3-ultrafast source tree"
	fi

	if [[ "$VERSION" != "latest" ]]; then
		expected_commit=$(
			/usr/bin/git -C "$SOURCE_DIR" rev-parse "refs/tags/${VERSION}^{commit}" 2>/dev/null || true
		)
		[[ -n "$expected_commit" ]] ||
			fail "release tag '$VERSION' does not exist in $SOURCE_DIR"
		actual_commit=$(/usr/bin/git -C "$SOURCE_DIR" rev-parse HEAD)
		[[ "$actual_commit" == "$expected_commit" ]] ||
			fail "$SOURCE_DIR is not checked out at requested release tag '$VERSION'"
	fi
}

build_from_source() {
	CURRENT_STEP="checking source-build tools"
	ensure_developer_tools
	ensure_go
	[[ "$(go env CGO_ENABLED)" == "1" ]] ||
		fail "CGO is disabled. Unset CGO_ENABLED or set CGO_ENABLED=1 and retry."
	prepare_source

	CURRENT_STEP="building the Metal engine from source"
	log "Building mumax3-ultrafast from source"
	/bin/mkdir -p "$INSTALL_DIR"
	STAGED_BINARY=$(/usr/bin/mktemp "${INSTALL_DIR%/}/.mumax3.new.XXXXXX")
	commit_hash=$(/usr/bin/git -C "$SOURCE_DIR" rev-parse --short HEAD 2>/dev/null || printf 'unknown')
	(
		cd "$SOURCE_DIR"
		MACOSX_DEPLOYMENT_TARGET=14.0 \
			CGO_CFLAGS='-mmacosx-version-min=14.0' \
			CGO_CXXFLAGS='-mmacosx-version-min=14.0' \
			CGO_LDFLAGS='-mmacosx-version-min=14.0' \
			CGO_ENABLED=1 go build -trimpath \
			-ldflags "-X main.commitHash=${commit_hash}" \
			-o "$STAGED_BINARY" ./cmd/mumax3
	)
}

verify_and_install_staged_binary() {
	[[ -n "$STAGED_BINARY" && -x "$STAGED_BINARY" ]] ||
		fail "the staged mumax3 executable is missing"

	CURRENT_STEP="checking desktop compatibility"
	bounded_engine_probe "$STAGED_BINARY" || true
	probe_output=$PROBE_RESULT
	[[ "$probe_output" == "mumax3-ultrafast desktop-api=1 backend=metal" ]] ||
		fail "the staged executable is not a compatible mumax3-ultrafast Metal engine"

	CURRENT_STEP="running the Metal smoke test"
	log "Verifying the Metal backend before installation"
	"$STAGED_BINARY" -test

	CURRENT_STEP="atomically installing the engine executable"
	/bin/mv -f "$STAGED_BINARY" "${INSTALL_DIR%/}/mumax3"
	STAGED_BINARY=""
}

uninstall_engine() {
	target="${INSTALL_DIR%/}/mumax3"
	if [[ -e "$target" ]]; then
		bounded_engine_probe "$target" || true
		uninstall_probe=$PROBE_RESULT
		[[ "$uninstall_probe" == "mumax3-ultrafast desktop-api=1 backend=metal" ]] ||
			fail "refusing to remove $target because it is not a compatible mumax3-ultrafast engine"
		/bin/rm -f "$target"
		log "Uninstalled mumax3-ultrafast"
		note "Removed: $target"
	else
		note "mumax3-ultrafast is not installed at $target"
	fi
	note "Any PATH line in ~/.zprofile was left unchanged because the directory may contain other commands."
}

main() {
	CURRENT_STEP="checking macOS compatibility"
	log "Checking this Mac"
	check_mac_compatibility

	case "$INSTALL_DIR" in
		/*) ;;
		*) INSTALL_DIR="$PWD/$INSTALL_DIR" ;;
	esac
	if ((UNINSTALL == 1)); then
		uninstall_engine
		return
	fi

	if ((FROM_SOURCE == 1)); then
		build_from_source
	elif release_available; then
		download_release
	else
		release_status=$?
		if ((release_status == 1)); then
			warn "No prebuilt ${VERSION} engine artifact is published yet; falling back to a source build."
			warn "The fallback may install Apple Command Line Tools, Homebrew, and Go."
			build_from_source
		else
			fail "could not check the engine release. Check the network and retry, or use --from-source explicitly."
		fi
	fi
	verify_and_install_staged_binary

	CURRENT_STEP="configuring the command path"
	ensure_profile_path
	export PATH="${INSTALL_DIR%/}:$PATH"
	hash -r

	CURRENT_STEP="complete"
	log "Installation complete"
	note "Executable: ${INSTALL_DIR%/}/mumax3"
	if ((NO_PROFILE == 0)); then
		note "Open a new Terminal (or run: source \"$profile_path\") before running: mumax3 example.mx3"
	else
		note "For this shell, add: export PATH=\"${INSTALL_DIR%/}:\$PATH\""
	fi
	note "The desktop app is optional and was not installed."
	note "Re-running this installer replaces mumax3 only after verification; use --uninstall to remove it."
}

main
