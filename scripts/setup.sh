#!/usr/bin/env bash
# One-command setup for the Intelligence Engine.
# Usage:  chmod +x setup.sh && ./setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Intelligence Engine v5.2 Setup ==="
echo "Root: $ROOT"
echo

# 1. Python virtual environment
if [ ! -d "$ROOT/venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv "$ROOT/venv"
else
    echo "[1/5] venv already exists, skipping"
fi
source "$ROOT/venv/bin/activate"

# 2. Dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q "$ROOT"

# 3. Data directory
echo "[3/5] Ensuring data directories..."
mkdir -p "$ROOT/intelligence/data" "$ROOT/logs"

# 4. Self-test
echo "[4/5] Running self-test..."
python3 -c "from intelligence.processing.selftest import run_selftest; r=run_selftest(); print(f'  {r[\"passed\"]}/{r[\"checks\"]} checks passed'); exit(0 if r['ok'] else 1)" || {
    echo "  SELF-TEST FAILED. The gate pipeline is broken. Fix before starting."
    exit 1
}
echo "  Self-test passed."

# 5. Benchmark
echo "[5/5] Running synthetic benchmark..."
python3 -m intelligence.validation.synthetic_benchmark 2>&1 | head -5

echo
echo "=== Setup complete ==="
echo "Start the engine:"
echo "  source venv/bin/activate"
echo "  python3 -m intelligence.main continuous"
echo "Then open: http://localhost:8000/dashboard"
echo
echo "Single-cycle run (no continuous mode):"
echo "  python3 -m intelligence.main once"
