#!/bin/bash
# Script to build and run the Dockerized test environment.
#
# Usage:
#   ./run_tests_docker.sh                    # default: Python 3.12
#   ./run_tests_docker.sh --python 3.10      # single version
#   ./run_tests_docker.sh --all-versions     # 3.8, 3.9, 3.10, 3.11, 3.12
#   ./run_tests_docker.sh --rebuild --python 3.9
#
# The --all-versions flag runs each version sequentially and prints a
# summary at the end so you can check everything locally before pushing.

set -o pipefail

cd "$(dirname "$0")"

IMAGE_NAME="quantpits-test-env"

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
PYTHON_VERSIONS=()
REBUILD_FLAG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_VERSIONS+=("$2"); shift 2 ;;
        --all-versions)
            PYTHON_VERSIONS=(3.8 3.9 3.10 3.11 3.12); shift ;;
        --rebuild)
            REBUILD_FLAG="--no-cache"; shift ;;
        -h|--help)
            echo "Usage: ./run_tests_docker.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --python X.Y       Test a single Python version (default: 3.12)"
            echo "  --all-versions     Test Python 3.8, 3.9, 3.10, 3.11, 3.12 sequentially"
            echo "  --rebuild          Force clean Docker rebuild (--no-cache)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_tests_docker.sh                         # Python 3.12"
            echo "  ./run_tests_docker.sh --python 3.10           # Python 3.10 only"
            echo "  ./run_tests_docker.sh --all-versions          # All 5 versions"
            echo "  ./run_tests_docker.sh --all-versions --rebuild"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Default to 3.12 if no version specified
if [ ${#PYTHON_VERSIONS[@]} -eq 0 ]; then
    PYTHON_VERSIONS=(3.12)
fi

# ------------------------------------------------------------------
# Run tests for each Python version
# ------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

for PY_VER in "${PYTHON_VERSIONS[@]}"; do
    TAG="${IMAGE_NAME}:py${PY_VER}"
    echo ""
    echo "============================================================================"
    echo "  Python $PY_VER  —  building image"
    echo "============================================================================"

    if ! docker build $REBUILD_FLAG \
        --build-arg PYTHON_VERSION="$PY_VER" \
        -f Dockerfile.test \
        -t "$TAG" \
        . ; then
        RESULTS["$PY_VER"]="BUILD_FAIL"
        ((FAIL_COUNT++))
        echo "--- Python $PY_VER : BUILD FAIL ---"
        continue
    fi

    echo ""
    echo "--- Running tests on Python $PY_VER ---"

    if docker run --rm "$TAG"; then
        RESULTS["$PY_VER"]="PASS"
        ((PASS_COUNT++))
        echo "--- Python $PY_VER : PASS ---"
    else
        RESULTS["$PY_VER"]="FAIL"
        ((FAIL_COUNT++))
        echo "--- Python $PY_VER : FAIL ---"
    fi
done

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "  TEST SUMMARY"
echo "============================================================================"
for PY_VER in "${PYTHON_VERSIONS[@]}"; do
    STATUS="${RESULTS[$PY_VER]}"
    case "$STATUS" in
        PASS)       echo "  Python $PY_VER :  PASS" ;;
        FAIL)       echo "  Python $PY_VER :  FAIL (tests)" ;;
        BUILD_FAIL) echo "  Python $PY_VER :  FAIL (docker build)" ;;
        *)          echo "  Python $PY_VER :  ???" ;;
    esac
done
echo "============================================================================"
echo "  $PASS_COUNT passed, $FAIL_COUNT failed out of ${#PYTHON_VERSIONS[@]} versions"
echo "============================================================================"

# Exit with non-zero if any version failed
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
