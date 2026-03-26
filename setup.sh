#!/usr/bin/env bash
# setup.sh — Cassandra project bootstrap
#
# USAGE:
#   First-time setup:       bash setup.sh
#   Auto-activate on cd:    Add the following to your ~/.zshrc or ~/.bashrc:
#
#       source ~/Documents/GitHub/Cassandra/setup.sh
#
#   Or, for any directory auto-activation, add this to ~/.zshrc:
#
#       function cd() { builtin cd "$@" && [ -f .venv/bin/activate ] && source .venv/bin/activate; }

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="python3.11"

# ── 1. Ensure Python 3.11 is available ──────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[setup] ERROR: $PYTHON not found. Install it first (e.g. brew install python@3.11)."
    exit 1
fi

# ── 2. Create the virtual environment if it doesn't exist ───────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "[setup] Virtual environment created."
else
    echo "[setup] Virtual environment already exists — skipping creation."
fi

# ── 3. Activate the virtual environment ─────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[setup] Virtual environment activated: $(which python)"

# ── 4. Upgrade pip silently ──────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 5. Install / sync requirements ──────────────────────────────────────────
if [ -f "$REPO_DIR/requirements.txt" ]; then
    echo "[setup] Installing dependencies from requirements.txt ..."
    pip install -r "$REPO_DIR/requirements.txt" --quiet
    echo "[setup] Dependencies installed."
else
    echo "[setup] WARNING: requirements.txt not found — skipping dependency install."
fi

# ── 6. Create .envrc for direnv users (auto-activation via direnv) ───────────
ENVRC="$REPO_DIR/.envrc"
if [ ! -f "$ENVRC" ]; then
    cat > "$ENVRC" <<'EOF'
# .envrc — direnv auto-activation
# Run `direnv allow` once to enable auto-activation on `cd`.
source .venv/bin/activate
EOF
    echo "[setup] Created .envrc for direnv. Run: direnv allow"
fi

# ── 7. Create .env from .env.example if not yet present ─────────────────────
ENV_FILE="$REPO_DIR/.env"
ENV_EXAMPLE="$REPO_DIR/.env.example"
if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "[setup] Created .env from .env.example — fill in your Binance testnet credentials."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cassandra environment ready."
echo "  Python: $(python --version)"
echo "  Venv:   $VENV_DIR"
echo ""
echo "  Next steps:"
echo "  1. Fill in .env with Binance testnet keys"
echo "     (https://testnet.binance.vision)"
echo ""
echo "  To auto-activate on every 'cd' into this"
echo "  directory, add this to your ~/.zshrc:"
echo ""
echo '    function cd() {'
echo '      builtin cd "$@"'
echo '      [ -f .venv/bin/activate ] && source .venv/bin/activate'
echo '    }'
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
