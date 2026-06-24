#!/bin/bash
# Create paper.zip for Overleaf upload
# This script packages all necessary files for Overleaf

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_NAME="paper.zip"
TEMP_DIR="/tmp/overleaf_upload_$$"

echo "📦 Creating $ZIP_NAME for Overleaf upload..."

# Clean up any existing temp dir
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Copy main files
echo "  Copying main.tex..."
cp main.tex "$TEMP_DIR/"

# Copy sections directory
echo "  Copying sections/..."
cp -r sections "$TEMP_DIR/"

# Copy figures directory
echo "  Copying figures/..."
cp -r figures "$TEMP_DIR/"

# Copy references.bib
echo "  Copying references.bib..."
cp references.bib "$TEMP_DIR/"

# Copy compile.sh
echo "  Copying compile.sh..."
cp compile.sh "$TEMP_DIR/"
chmod +x "$TEMP_DIR/compile.sh"

# Create the zip
echo "  Creating zip archive..."
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$ZIP_NAME" . -x "*.aux" "*.log" "*.bbl" "*.blg" "*.out"
cd "$SCRIPT_DIR"

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ Created $ZIP_NAME"
echo "📁 Contents:"
unzip -l "$ZIP_NAME" | grep -v "^Archive:" | grep -v "^  Length" | grep -v "^---------"
echo ""
echo "🚀 Upload to Overleaf: https://www.overleaf.com/project"
