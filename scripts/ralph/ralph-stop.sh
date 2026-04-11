#!/bin/bash
# Ralph Loop - Stop/Cancel Script
set -euo pipefail

# Ensure Git Bash / MSYS coreutils are in PATH
if [ -d "/usr/bin" ]; then
  export PATH="/usr/bin:/bin:${PATH}"
fi
if [ -d "/c/Program Files/Git/usr/bin" ]; then
  export PATH="/c/Program Files/Git/usr/bin:${PATH}"
fi

FORCE=false
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
  FORCE=true
fi

echo "Checking for active Ralph loops..."

# Check for fresh-context processes
RALPH_PIDS=$(pgrep -f "ralph.sh" 2>/dev/null || echo "")
CLAUDE_PIDS=$(pgrep -f "claude --print.*ralph" 2>/dev/null || echo "")

# Check for same-session state
SAME_SESSION=false
if [ -f ".claude/ralph-loop.local.md" ]; then
  SAME_SESSION=true
fi

FRESH_CONTEXT=false
if [ -n "$RALPH_PIDS" ]; then
  FRESH_CONTEXT=true
fi

if [ "$SAME_SESSION" = false ] && [ "$FRESH_CONTEXT" = false ]; then
  echo "No active Ralph loops found."
  exit 0
fi

if [ "$SAME_SESSION" = true ]; then
  echo "Found: same-session loop (.claude/ralph-loop.local.md)"
fi
if [ "$FRESH_CONTEXT" = true ]; then
  echo "Found: fresh-context processes (PIDs: $RALPH_PIDS)"
fi

if [ "$FORCE" = true ]; then
  # Kill processes
  if [ -n "$RALPH_PIDS" ]; then
    echo "Killing ralph.sh processes..."
    kill $RALPH_PIDS 2>/dev/null || true
  fi
  if [ -n "$CLAUDE_PIDS" ]; then
    echo "Killing Claude subprocesses..."
    kill $CLAUDE_PIDS 2>/dev/null || true
  fi
  # Remove state files
  rm -f .claude/ralph-loop.local.md
  rm -f .claude/ralph-state.local.md
  rm -f .claude/ralph-status.local.json
  echo "All Ralph loops stopped."
else
  echo ""
  echo "Run with --force to stop: ./scripts/ralph/ralph-stop.sh --force"
fi
