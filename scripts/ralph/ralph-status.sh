#!/bin/bash
# Ralph Status Dashboard — SBTDD Edition
# Adapted from ralph-loop-setup v1.4.0 template for tasks[]+status format
#
# Usage:
#   ./scripts/ralph/ralph-status.sh          # One-time status
#   ./scripts/ralph/ralph-status.sh --watch  # Live updates (every 2s)

set -euo pipefail

# Ensure Git Bash / MSYS coreutils are in PATH (Windows compat)
if [ -d "/usr/bin" ]; then
  export PATH="/usr/bin:/bin:${PATH}"
fi
if [ -d "/c/Program Files/Git/usr/bin" ]; then
  export PATH="/c/Program Files/Git/usr/bin:${PATH}"
fi

STATUS_FILE=".claude/ralph-status.local.json"
PRD_FILE="plans/prd.json"
PROGRESS_FILE="plans/progress.md"
RUNS_DIR="scripts/ralph/runs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
WATCH_MODE=false
if [ "${1:-}" = "--watch" ] || [ "${1:-}" = "-w" ]; then
  WATCH_MODE=true
fi

print_header() {
  echo ""
  echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║         🤖 Ralph Loop Status Dashboard (SBTDD)               ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

print_no_loop() {
  echo -e "${YELLOW}No active Ralph loop detected.${NC}"
  echo ""
  echo "To start a loop:"
  echo "  ./scripts/ralph/ralph.sh --verbose"
  echo "  ./scripts/ralph/ralph.sh --branch ralph/feature --verbose"
  echo ""
}

print_status() {
  if [ ! -f "$STATUS_FILE" ]; then
    print_no_loop
    return
  fi

  RUN_ID=$(jq -r '.run_id // "unknown"' "$STATUS_FILE")
  ITERATION=$(jq -r '.iteration // 0' "$STATUS_FILE")
  MAX_ITER=$(jq -r '.max_iterations // 0' "$STATUS_FILE")
  STATUS=$(jq -r '.status // "unknown"' "$STATUS_FILE")
  TASK_ID=$(jq -r '.current_task.id // "-"' "$STATUS_FILE")
  TASK_TITLE=$(jq -r '.current_task.title // "-"' "$STATUS_FILE")
  REMAINING=$(jq -r '.remaining_tasks // 0' "$STATUS_FILE")
  STARTED=$(jq -r '.started_at // ""' "$STATUS_FILE")
  UPDATED=$(jq -r '.updated_at // ""' "$STATUS_FILE")
  BRANCH=$(jq -r '.branch // "-"' "$STATUS_FILE")

  case "$STATUS" in
    running)        STATUS_COLOR="${GREEN}● RUNNING${NC}" ;;
    verifying)      STATUS_COLOR="${CYAN}◐ VERIFYING${NC}" ;;
    complete)       STATUS_COLOR="${GREEN}✓ COMPLETE${NC}" ;;
    starting)       STATUS_COLOR="${YELLOW}○ STARTING${NC}" ;;
    max_iterations) STATUS_COLOR="${RED}✗ MAX ITERATIONS${NC}" ;;
    *)              STATUS_COLOR="${YELLOW}? $STATUS${NC}" ;;
  esac

  # Duration (uses GNU date from Git Bash)
  START_EPOCH=$(date -d "$STARTED" +%s 2>/dev/null || echo 0)
  NOW_EPOCH=$(date +%s)
  if [ "$START_EPOCH" -gt 0 ]; then
    DURATION=$((NOW_EPOCH - START_EPOCH))
    DURATION_STR="$(( DURATION / 60 ))m $(( DURATION % 60 ))s"
  else
    DURATION_STR="--"
  fi

  echo -e "${BOLD}Loop Status:${NC} $STATUS_COLOR"
  echo -e "${BOLD}Run ID:${NC}      $RUN_ID"
  echo -e "${BOLD}Branch:${NC}      $BRANCH"
  echo -e "${BOLD}Duration:${NC}    $DURATION_STR"
  echo ""
  echo -e "${BOLD}Progress:${NC}    Iteration $ITERATION of $MAX_ITER"

  if [ "$MAX_ITER" -gt 0 ]; then
    PCT=$((ITERATION * 100 / MAX_ITER))
    BAR_WIDTH=40
    FILLED=$((PCT * BAR_WIDTH / 100))
    EMPTY=$((BAR_WIDTH - FILLED))
    printf "             ["
    printf "%0.s█" $(seq 1 $FILLED 2>/dev/null) || true
    printf "%0.s░" $(seq 1 $EMPTY 2>/dev/null) || true
    printf "] %d%%\n" $PCT
  fi

  echo ""
  echo -e "${BOLD}Current Task:${NC}"
  echo -e "  ID:    $TASK_ID"
  echo -e "  Title: $TASK_TITLE"
  echo ""
  echo -e "${BOLD}Tasks Remaining:${NC} $REMAINING"
}

print_tasks() {
  if [ ! -f "$PRD_FILE" ]; then
    echo -e "${YELLOW}No $PRD_FILE found${NC}"
    return
  fi

  echo ""
  echo -e "${BOLD}Task List (SBTDD):${NC}"
  echo "─────────────────────────────────────────────────────────────────"

  jq -r '.tasks[] |
    if .status == "completed" then
      "  ✓ \(.id) - \(.title)"
    elif .status == "in_progress" then
      "  ◐ \(.id) - \(.title) [IN PROGRESS]"
    elif .status == "blocked" then
      "  ⊘ \(.id) - \(.title) [BLOCKED]"
    else
      "  ○ \(.id) - \(.title)"
    end
  ' "$PRD_FILE"

  echo "─────────────────────────────────────────────────────────────────"

  DONE=$(jq '[.tasks[] | select(.status == "completed")] | length' "$PRD_FILE")
  BLOCKED=$(jq '[.tasks[] | select(.status == "blocked")] | length' "$PRD_FILE")
  TOTAL=$(jq '.tasks | length' "$PRD_FILE")
  if [ "$BLOCKED" -gt 0 ]; then
    echo -e "  ${GREEN}$DONE${NC} of ${BOLD}$TOTAL${NC} complete, ${RED}$BLOCKED${NC} blocked"
  else
    echo -e "  ${GREEN}$DONE${NC} of ${BOLD}$TOTAL${NC} tasks complete"
  fi
}

print_recent_runs() {
  echo ""
  echo -e "${BOLD}Recent Runs:${NC}"

  if [ ! -d "$RUNS_DIR" ]; then
    echo "  No runs yet"
    return
  fi

  RUNS=$(ls -t "$RUNS_DIR" 2>/dev/null | head -5)

  if [ -z "$RUNS" ]; then
    echo "  No runs yet"
    return
  fi

  for RUN in $RUNS; do
    ITER_COUNT=$(ls "$RUNS_DIR/$RUN"/iteration-*.txt 2>/dev/null | wc -l | tr -d ' ')
    echo "  $RUN - $ITER_COUNT iterations"
  done
}

print_log_tail() {
  if [ ! -f "$STATUS_FILE" ]; then
    return
  fi

  LOG_FILE=$(jq -r '.log_file // ""' "$STATUS_FILE")

  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo ""
    echo -e "${BOLD}Latest Output:${NC} (last 8 lines)"
    echo "─────────────────────────────────────────────────────────────────"
    tail -8 "$LOG_FILE" | sed 's/^/  /'
    echo "─────────────────────────────────────────────────────────────────"
  fi
}

# Main
if [ "$WATCH_MODE" = true ]; then
  while true; do
    clear
    print_header
    print_status
    print_tasks
    print_log_tail
    echo ""
    echo -e "${CYAN}Refreshing every 2s... (Ctrl+C to exit)${NC}"
    sleep 2
  done
else
  print_header
  print_status
  print_tasks
  print_recent_runs
  print_log_tail
fi
