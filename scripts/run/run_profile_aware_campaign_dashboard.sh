#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
One-click workflow:
  1) campaign compare-profiles
  2) campaign render-dashboard
  3) open dashboard html

Usage:
  bash scripts/run_profile_aware_campaign_dashboard.sh [options]

Options:
  --source <example|campaign>              Data source mode (default: example)
  --campaign-config <path>                 Required when --source campaign
  --output-root-dir <dir>                  Comparison output root
  --output-html <path>                     Dashboard HTML output path
  --pair-mode <adjacent|all_pairs>         Profile pair mode (default: adjacent)
  --artifact-hint-path-mode <relative|absolute>
                                           Hint path mode (default: relative)
  --profiles <p1> <p2> [p3 ...]            Profiles list
  --uv-cache-dir <dir>                     UV cache directory (default: /tmp/uv-cache)
  --no-open                                Generate and render only, do not open HTML
  -h, --help                               Show help
EOF
}

SOURCE="example"
CAMPAIGN_CONFIG=""
OUTPUT_ROOT_DIR="dist/examples/profile_aware_campaign_level12"
OUTPUT_HTML=""
PAIR_MODE="adjacent"
ARTIFACT_HINT_PATH_MODE="relative"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-/tmp/uv-cache}"
NO_OPEN=0
PROFILES=("exploratory_screening" "default_research" "stricter_research")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --campaign-config)
            CAMPAIGN_CONFIG="$2"
            shift 2
            ;;
        --output-root-dir)
            OUTPUT_ROOT_DIR="$2"
            shift 2
            ;;
        --output-html)
            OUTPUT_HTML="$2"
            shift 2
            ;;
        --pair-mode)
            PAIR_MODE="$2"
            shift 2
            ;;
        --artifact-hint-path-mode)
            ARTIFACT_HINT_PATH_MODE="$2"
            shift 2
            ;;
        --profiles)
            shift
            PROFILES=()
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                PROFILES+=("$1")
                shift
            done
            ;;
        --uv-cache-dir)
            UV_CACHE_DIR_VALUE="$2"
            shift 2
            ;;
        --no-open)
            NO_OPEN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$SOURCE" != "example" && "$SOURCE" != "campaign" ]]; then
    echo "--source must be one of: example, campaign" >&2
    exit 1
fi

if [[ ${#PROFILES[@]} -lt 2 ]]; then
    echo "--profiles requires at least two profile names" >&2
    exit 1
fi

if [[ "$SOURCE" == "campaign" && -z "$CAMPAIGN_CONFIG" ]]; then
    echo "--campaign-config is required when --source campaign" >&2
    exit 1
fi

if [[ -z "$OUTPUT_HTML" ]]; then
    OUTPUT_HTML="${OUTPUT_ROOT_DIR}/campaign_profile_dashboard_zh.html"
fi

COMPARISON_JSON="${OUTPUT_ROOT_DIR}/campaign_profile_comparison.json"

open_file() {
    local file_path="$1"
    local abs_path="$file_path"

    if command -v realpath >/dev/null 2>&1; then
        abs_path="$(realpath "$file_path")"
    fi

    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$abs_path" >/dev/null 2>&1 && return 0
    fi

    if command -v wslview >/dev/null 2>&1; then
        wslview "$abs_path" >/dev/null 2>&1 && return 0
    fi

    if command -v powershell.exe >/dev/null 2>&1; then
        local windows_path="$abs_path"
        if command -v wslpath >/dev/null 2>&1; then
            windows_path="$(wslpath -w "$abs_path")"
        fi
        powershell.exe -NoProfile -Command "Start-Process -FilePath '$windows_path'" >/dev/null 2>&1 && return 0
    fi

    if command -v open >/dev/null 2>&1; then
        open "$abs_path" >/dev/null 2>&1 && return 0
    fi

    return 1
}

compare_cmd=(
    uv run --no-sync --frozen alpha-lab campaign compare-profiles
    --source "$SOURCE"
    --output-root-dir "$OUTPUT_ROOT_DIR"
    --pair-mode "$PAIR_MODE"
    --artifact-hint-path-mode "$ARTIFACT_HINT_PATH_MODE"
    --profiles "${PROFILES[@]}"
)

if [[ "$SOURCE" == "campaign" ]]; then
    compare_cmd+=(--campaign-config "$CAMPAIGN_CONFIG")
fi

render_cmd=(
    uv run --no-sync --frozen alpha-lab campaign render-dashboard
    --comparison-json "$COMPARISON_JSON"
    --output-html "$OUTPUT_HTML"
    --overwrite
)

echo "[1/3] Running campaign profile comparison..."
UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" "${compare_cmd[@]}"

echo "[2/3] Rendering dashboard..."
UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" "${render_cmd[@]}"

if [[ "$NO_OPEN" -eq 1 ]]; then
    echo "[3/3] Skipped opening dashboard (--no-open)."
    echo "Dashboard: $OUTPUT_HTML"
    exit 0
fi

echo "[3/3] Opening dashboard..."
if open_file "$OUTPUT_HTML"; then
    echo "Dashboard opened: $OUTPUT_HTML"
else
    echo "Dashboard generated, but auto-open failed."
    echo "Open manually: $OUTPUT_HTML"
fi
