#!/bin/bash
# Watch paper directory for changes and auto-create zip
# Usage: ./watch_and_zip.sh [--once]
#   --once: Create zip once and exit (no watching)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_NAME="paper.zip"
LAST_HASH=""

# Function to calculate hash of relevant files
calculate_hash() {
    find . -type f \( -name "*.tex" -o -name "*.bib" -o -name "*.sty" \) \
        -not -path "./.git/*" \
        -not -name "$ZIP_NAME" \
        -exec md5 -q {} \; 2>/dev/null | sort | md5 -q
}

# Function to create zip
create_zip() {
    ./create_zip.sh
    LAST_HASH=$(calculate_hash)
}

# Check for --once flag
if [[ "$1" == "--once" ]]; then
    create_zip
    exit 0
fi

# Check if fswatch is installed
if ! command -v fswatch &> /dev/null; then
    echo "⚠️  fswatch not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install fswatch
    else
        echo "❌ Please install fswatch manually:"
        echo "   brew install fswatch"
        echo ""
        echo "   Or run with --once flag: ./watch_and_zip.sh --once"
        exit 1
    fi
fi

echo "👀 Watching paper directory for changes..."
echo "   Press Ctrl+C to stop"
echo ""

# Initial zip creation
create_zip

# Watch for changes
fswatch -0 --event Updated --event Created --event Removed \
    --exclude '\.zip$' \
    --exclude '\.aux$' \
    --exclude '\.log$' \
    --exclude '\.bbl$' \
    --exclude '\.blg$' \
    --exclude '\.out$' \
    --exclude '\.git' \
    . | while read -d "" event; do
    
    # Calculate new hash
    NEW_HASH=$(calculate_hash)
    
    # Only recreate if hash changed
    if [[ "$NEW_HASH" != "$LAST_HASH" ]]; then
        echo ""
        echo "📝 Change detected: $event"
        create_zip
        echo "✅ Updated $ZIP_NAME"
    fi
done
