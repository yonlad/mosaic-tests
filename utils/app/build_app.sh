#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Blend Review Tool — build ==="

# Ensure PyInstaller is available
if ! python3 -m PyInstaller --version &>/dev/null; then
    echo "Installing build dependencies..."
    pip3 install -r requirements.txt
fi

echo "Running PyInstaller..."
python3 -m PyInstaller \
    --noconfirm \
    --windowed \
    --name "Blend Review" \
    --target-arch arm64 \
    --collect-data botocore \
    --collect-data boto3 \
    --collect-data certifi \
    --hidden-import review \
    --hidden-import review_blends \
    --add-data ".env:." \
    launcher.py

echo ""
echo "=== Build complete ==="
echo "App: $SCRIPT_DIR/dist/Blend Review.app"
echo ""
echo "To ship to your colleague:"
echo "  1. Create a folder, e.g. 'BlendReview'"
echo "  2. Copy 'dist/Blend Review.app' into it"
echo "  3. Copy '.env' into the same folder (next to the .app)"
echo "  4. Zip and AirDrop / email the folder"
echo ""
echo "First-run on his Mac: right-click the app -> Open -> Open (Gatekeeper)"
