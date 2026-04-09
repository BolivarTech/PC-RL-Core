#!/bin/bash
# Ralph Loop - Fresh Context Mode
# Customized for pc-rl-core SBTDD workflow
# Uses tasks[] with status field (not features[] with passes)

set -euo pipefail

# Ensure Git Bash / MSYS coreutils are in PATH when launched from
# PowerShell/cmd where the Windows PATH may not include /usr/bin.
# This guarantees date, sed, grep, awk, jq, etc. are findable.
if [ -d "/usr/bin" ]; then
  export PATH="/usr/bin:/bin:${PATH}"
fi
if [ -d "/c/Program Files/Git/usr/bin" ]; then
  export PATH="/c/Program Files/Git/usr/bin:${PATH}"
fi

# ============================================
# PROJECT-SPECIFIC CONFIGURATION
# ============================================

VERIFY_COMMAND="python run-tests.py"
PRD_FILE="plans/prd.json"
PROGRESS_FILE="plans/progress.md"
GUARDRAILS_FILE="plans/guardrails.md"
PROMPT_FILE=".claude/ralph/PROMPT.md"
SPEC_FILE="sbtdd/spec-behavior.md"

# ============================================
# Configuration
# ============================================

MAX_ITERATIONS=100
BRANCH=""
VERBOSE=false
STATE_FILE=".claude/ralph-state.local.md"
STATUS_FILE=".claude/ralph-status.local.json"
RUNS_DIR="scripts/ralph/runs"
PROJECT_DIR="$(pwd)"

# ============================================
# Parse Arguments
# ============================================

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./scripts/ralph/ralph.sh [--max-iterations N] [--branch NAME] [--verbose|-v]"
      exit 1
      ;;
  esac
done

# ============================================
# Helper Functions
# ============================================

log_verbose() {
  if [ "$VERBOSE" = true ]; then
    echo "[VERBOSE] $*"
  fi
}

# Write status file for ralph-status.sh dashboard
# Args: $1=status ($2=task_id, $3=task_title, $4=remaining, $5=log_file)
update_status() {
  local status="$1"
  local task_id="${2:-}"
  local task_title="${3:-}"
  local remaining="${4:-0}"
  local log_file="${5:-}"

  mkdir -p "$(dirname "$STATUS_FILE")"
  jq -n \
    --arg run_id "$RUN_ID" \
    --argjson iteration "$ITERATION" \
    --argjson max_iterations "$MAX_ITERATIONS" \
    --arg status "$status" \
    --arg task_id "$task_id" \
    --arg task_title "$task_title" \
    --argjson remaining "$remaining" \
    --arg started_at "$START_TIME" \
    --arg updated_at "$(date -Iseconds 2>/dev/null || date)" \
    --arg branch "${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')}" \
    --arg log_file "$log_file" \
    '{
      run_id: $run_id,
      iteration: $iteration,
      max_iterations: $max_iterations,
      status: $status,
      current_task: { id: $task_id, title: $task_title },
      remaining_tasks: $remaining,
      started_at: $started_at,
      updated_at: $updated_at,
      branch: $branch,
      log_file: $log_file
    }' > "$STATUS_FILE"
}

# ============================================
# Setup
# ============================================

RUN_ID=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$RUNS_DIR/$RUN_ID"
START_TIME=$(date -Iseconds 2>/dev/null || date)
ITERATION=0
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Ralph Loop - Fresh Context Mode (SBTDD)"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Branch: ${BRANCH:-<current>}"
echo "Verify: $VERIFY_COMMAND"
echo "PRD: $PRD_FILE"
echo "Spec: $SPEC_FILE"
echo "Logs: $RUN_DIR/"
echo "=============================================="
echo ""

# Handle branch
if [ -n "$BRANCH" ]; then
  if git branch --list "$BRANCH" | grep -q "$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH"
  fi
  echo "Working on branch: $BRANCH"
fi

# Check prd.json exists
if [ ! -f "$PRD_FILE" ]; then
  echo "Error: $PRD_FILE not found"
  exit 1
fi

# Count pending tasks (uses tasks[] with status)
REMAINING_TASKS=$(jq '[.tasks[] | select(.status == "pending" or .status == "in_progress")] | length' "$PRD_FILE")
if [ "$REMAINING_TASKS" -eq 0 ]; then
  echo "All tasks in plans/prd.json are already complete!"
  exit 0
fi

echo "Found $REMAINING_TASKS pending tasks"
echo ""

update_status "starting" "" "" "$REMAINING_TASKS" ""

# ============================================
# Main Loop
# ============================================

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
  ITERATION=$((ITERATION + 1))

  echo ""
  echo "=============================================="
  echo "Iteration $ITERATION of $MAX_ITERATIONS"
  echo "=============================================="

  # Get next pending task (by id order)
  NEXT_TASK_JSON=$(jq '
    .tasks
    | map(select(.status == "pending" or .status == "in_progress"))
    | sort_by(.id)
    | first
  ' "$PRD_FILE")

  TASK_ID=$(echo "$NEXT_TASK_JSON" | jq -r '.id // "unknown"')
  TASK_TITLE=$(echo "$NEXT_TASK_JSON" | jq -r '.title // "unknown"')
  SECTION_REF=$(echo "$NEXT_TASK_JSON" | jq -r '.section_ref // ""')

  if [ "$TASK_ID" = "null" ] || [ "$TASK_ID" = "unknown" ]; then
    echo "All tasks complete!"
    rm -f "$STATE_FILE"
    exit 0
  fi

  echo "Task: $TASK_ID - $TASK_TITLE"
  echo "Section: $SECTION_REF"
  echo ""

  update_status "running" "$TASK_ID" "$TASK_TITLE" "$REMAINING_TASKS" "$RUN_DIR/iteration-$ITERATION.txt"

  log_verbose "Starting task $TASK_ID at $(date -Iseconds 2>/dev/null || date)"

  # Read guardrails
  GUARDRAILS_CONTENT=""
  if [ -f "$GUARDRAILS_FILE" ]; then
    GUARDRAILS_CONTENT=$(cat "$GUARDRAILS_FILE")
  fi

  # Create state file
  cat > "$STATE_FILE" << EOF
---
iteration: $ITERATION
max_iterations: $MAX_ITERATIONS
run_id: "$RUN_ID"
mode: fresh
---

## Guardrails (Signs)

$GUARDRAILS_CONTENT

---

## Instructions

You are in a Ralph loop (fresh-context mode) following SBTDD methodology.

1. Read $SPEC_FILE for the full SDD+BDD context
2. Read $PRD_FILE and find the first task where status is "pending"
3. Read the section_ref file for the specific TDD stubs
4. Read $PROGRESS_FILE for context from previous iterations
5. Follow strict TDD Red-Green-Refactor:
   - RED: write the tests from the section stubs. Run python run-tests.py. All new tests must FAIL.
   - GREEN: implement the minimum code to make tests pass. Run python run-tests.py. All tests must PASS.
   - REFACTOR: improve code quality without changing behavior. Run python run-tests.py. Still all PASS.
6. When the current task is complete:
   - Commit with message: "test: [task-id] - tests" then "feat: [task-id] - implementation"
   - Update plans/prd.json: set status to "completed"
   - Update $PROGRESS_FILE with learnings
7. After completing ONE task, check plans/prd.json:
   - If ALL tasks are "completed": output <promise>COMPLETE</promise>
   - If tasks remain: EXIT immediately (loop spawns fresh session for next task)

**Critical:** Complete ONE task, commit, then EXIT. Do NOT work on multiple tasks.
If a test passes without implementation (false positive), stop and report it.
If behavior is ambiguous, consult $SPEC_FILE before assuming.
Do not implement anything not covered by an existing test.
EOF

  # Spawn fresh Claude session
  echo "Spawning fresh Claude session..."
  OUTPUT_FILE="$RUN_DIR/iteration-$ITERATION.txt"
  JSON_FILE="$RUN_DIR/iteration-$ITERATION.json"

  log_verbose "Output: $OUTPUT_FILE"

  claude --print --output-format json --dangerously-skip-permissions \
    "You are in a Ralph loop (fresh-context mode, SBTDD methodology). \
     Read .claude/ralph-state.local.md for instructions. \
     Read $PRD_FILE to find the next pending task. \
     Read the task's section_ref for TDD stubs. \
     Read $SPEC_FILE for SDD+BDD context. \
     Follow strict TDD: test first, minimal impl, refactor. \
     Run 'python run-tests.py' after each change. \
     Complete ONE task only, commit, update plans/prd.json status to completed. \
     Output <promise>COMPLETE</promise> only when ALL tasks are completed." \
    > "$JSON_FILE" 2>&1 || true

  # Extract result text
  jq -r '.result // empty' "$JSON_FILE" > "$OUTPUT_FILE" 2>/dev/null || cp "$JSON_FILE" "$OUTPUT_FILE"

  if [ "$VERBOSE" = true ]; then
    OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
    echo "[VERBOSE] Output: $OUTPUT_LINES lines"
    echo "[VERBOSE] Last 10 lines:"
    tail -10 "$OUTPUT_FILE" | sed 's/^/  | /'
  fi

  # Check for completion promise
  if tail -10 "$OUTPUT_FILE" | grep -qE "<promise>COMPLETE</promise>"; then
    echo ""
    echo "=============================================="
    echo "COMPLETE — All tasks done!"
    echo "=============================================="
    rm -f "$STATE_FILE"
    exit 0
  fi

  # Run verification
  echo ""
  echo "Running verification..."
  update_status "verifying" "$TASK_ID" "$TASK_TITLE" "$REMAINING_TASKS" "$RUN_DIR/iteration-$ITERATION.txt"
  VERIFY_OUTPUT=$($VERIFY_COMMAND 2>&1) || true
  VERIFY_EXIT=$?

  if [ $VERIFY_EXIT -eq 0 ]; then
    echo "Verification PASSED"
  else
    echo "Verification FAILED (exit $VERIFY_EXIT)"
    echo "$VERIFY_OUTPUT" | tail -20
  fi

  # Check remaining tasks
  REMAINING_TASKS=$(jq '[.tasks[] | select(.status == "pending" or .status == "in_progress")] | length' "$PRD_FILE")
  echo "Remaining tasks: $REMAINING_TASKS"

  if [ "$REMAINING_TASKS" -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "All tasks complete!"
    echo "=============================================="
    update_status "complete" "" "" 0 ""
    rm -f "$STATE_FILE"
    exit 0
  fi

  sleep 2
done

echo ""
echo "=============================================="
echo "Max iterations ($MAX_ITERATIONS) reached"
echo "Remaining tasks: $REMAINING_TASKS"
echo "=============================================="
update_status "max_iterations" "$TASK_ID" "$TASK_TITLE" "$REMAINING_TASKS" "$RUN_DIR/iteration-$ITERATION.txt"
rm -f "$STATE_FILE"
exit 1
