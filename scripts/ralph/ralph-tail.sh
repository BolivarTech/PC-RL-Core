#!/bin/bash
# Ralph Log Tail — follow iteration output
#
# Usage:
#   ./scripts/ralph/ralph-tail.sh           # Follow current/latest run
#   ./scripts/ralph/ralph-tail.sh --all     # Follow all iteration files
#   ./scripts/ralph/ralph-tail.sh RUN_ID    # Follow specific run

set -euo pipefail

# Ensure Git Bash / MSYS coreutils are in PATH
if [ -d "/usr/bin" ]; then
  export PATH="/usr/bin:/bin:${PATH}"
fi
if [ -d "/c/Program Files/Git/usr/bin" ]; then
  export PATH="/c/Program Files/Git/usr/bin:${PATH}"
fi

STATUS_FILE=".claude/ralph-status.local.json"
RUNS_DIR="scripts/ralph/runs"

YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

get_run_dir() {
  if [ -f "$STATUS_FILE" ]; then
    RUN_ID=$(jq -r '.run_id' "$STATUS_FILE")
    if [ -d "$RUNS_DIR/$RUN_ID" ]; then
      echo "$RUNS_DIR/$RUN_ID"
      return
    fi
  fi

  LATEST=$(ls -t "$RUNS_DIR" 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "$RUNS_DIR/$LATEST"
  else
    echo ""
  fi
}

MODE="latest"
RUN_ID=""

if [ "${1:-}" = "--all" ] || [ "${1:-}" = "-a" ]; then
  MODE="all"
elif [ -n "${1:-}" ]; then
  RUN_ID="$1"
  MODE="specific"
fi

echo -e "${BOLD}Ralph Log Tail (SBTDD)${NC}"
echo ""

case "$MODE" in
  latest)
    RUN_DIR=$(get_run_dir)
    if [ -z "$RUN_DIR" ]; then
      echo -e "${YELLOW}No Ralph runs found in $RUNS_DIR${NC}"
      exit 1
    fi
    echo -e "Following: ${CYAN}$RUN_DIR${NC}"
    echo -e "Use Ctrl+C to exit"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""

    while true; do
      LATEST_FILE=$(ls -t "$RUN_DIR"/iteration-*.txt 2>/dev/null | head -1)
      if [ -n "$LATEST_FILE" ]; then
        timeout 5 tail -f "$LATEST_FILE" 2>/dev/null || true
      else
        echo "Waiting for iteration files..."
        sleep 2
      fi
    done
    ;;

  all)
    RUN_DIR=$(get_run_dir)
    if [ -z "$RUN_DIR" ]; then
      echo -e "${YELLOW}No Ralph runs found in $RUNS_DIR${NC}"
      exit 1
    fi
    echo -e "Following all iterations in: ${CYAN}$RUN_DIR${NC}"
    echo -e "Use Ctrl+C to exit"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""

    tail -F "$RUN_DIR"/iteration-*.txt 2>/dev/null || {
      echo "Waiting for iteration files..."
      sleep 2
      exec "$0" --all
    }
    ;;

  specific)
    if [ ! -d "$RUNS_DIR/$RUN_ID" ]; then
      echo -e "${YELLOW}Run not found: $RUNS_DIR/$RUN_ID${NC}"
      echo ""
      echo "Available runs:"
      ls -t "$RUNS_DIR" 2>/dev/null | head -10 | sed 's/^/  /'
      exit 1
    fi
    echo -e "Following: ${CYAN}$RUNS_DIR/$RUN_ID${NC}"
    echo -e "Use Ctrl+C to exit"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    tail -F "$RUNS_DIR/$RUN_ID"/iteration-*.txt 2>/dev/null
    ;;
esac
