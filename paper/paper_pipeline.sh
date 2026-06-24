#!/bin/bash
# Paper Pipeline - Complete workflow for paper updates
# Usage:
#   ./paper_pipeline.sh              # Create zip and watch for changes
#   ./paper_pipeline.sh --once       # Create zip once and exit
#   ./paper_pipeline.sh --watch      # Watch for changes only
#   ./paper_pipeline.sh --compile    # Compile paper first, then create zip
#   ./paper_pipeline.sh --help       # Show this help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Show help
show_help() {
    echo "Paper Pipeline - Complete workflow for paper updates"
    echo ""
    echo "Usage:"
    echo "  ./paper_pipeline.sh              # Create zip and watch for changes"
    echo "  ./paper_pipeline.sh --once       # Create zip once and exit"
    echo "  ./paper_pipeline.sh --watch      # Watch for changes only"
    echo "  ./paper_pipeline.sh --compile    # Compile paper first, then create zip"
    echo "  ./paper_pipeline.sh --help       # Show this help"
    echo ""
    echo "Features:"
    echo "  - Creates paper.zip with all files needed for Overleaf"
    echo "  - Watches for file changes and auto-updates zip"
    echo "  - Optional: Compile paper before creating zip"
    echo "  - Excludes auxiliary files (.aux, .log, .bbl, etc.)"
    echo ""
    echo "Overleaf Upload:"
    echo "  1. Run this script: ./paper_pipeline.sh"
    echo "  2. Go to https://www.overleaf.com/project"
    echo "  3. Click 'Upload' and select paper.zip"
    echo "  4. Overleaf will extract and compile automatically"
}

# Compile paper
compile_paper() {
    log_info "Compiling paper..."
    
    if [[ ! -f "compile.sh" ]]; then
        log_error "compile.sh not found"
        return 1
    fi
    
    chmod +x compile.sh
    if ./compile.sh; then
        log_success "Paper compiled successfully"
        return 0
    else
        log_warning "Compilation had warnings, but continuing..."
        return 0
    fi
}

# Create zip file
create_zip() {
    log_info "Creating paper.zip..."
    
    ZIP_NAME="paper.zip"
    TEMP_DIR="/tmp/overleaf_upload_$$"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    
    # Copy files
    cp main.tex "$TEMP_DIR/"
    cp -r sections "$TEMP_DIR/"
    cp -r figures "$TEMP_DIR/"
    cp references.bib "$TEMP_DIR/"
    cp compile.sh "$TEMP_DIR/"
    chmod +x "$TEMP_DIR/compile.sh"
    
    # Create zip
    cd "$TEMP_DIR"
    zip -r "$SCRIPT_DIR/$ZIP_NAME" . -x "*.aux" "*.log" "*.bbl" "*.blg" "*.out" > /dev/null 2>&1
    cd "$SCRIPT_DIR"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    log_success "Created $ZIP_NAME"
    
    # Show contents
    echo ""
    echo "📦 ZIP Contents:"
    unzip -l "$ZIP_NAME" | grep -v "^Archive:" | grep -v "^  Length" | grep -v "^---------"
    echo ""
}

# Watch for changes
watch_changes() {
    log_info "Watching for changes in paper directory..."
    echo "   Press Ctrl+C to stop"
    echo ""
    
    LAST_HASH=""
    
    # Function to calculate hash
    calculate_hash() {
        find . -type f \( -name "*.tex" -o -name "*.bib" -o -name "*.sty" \) \
            -not -path "./.git/*" \
            -not -name "*.zip" \
            -exec md5 -q {} \; 2>/dev/null | sort | md5 -q
    }
    
    # Initial hash
    LAST_HASH=$(calculate_hash)
    
    # Watch loop
    while true; do
        sleep 2
        
        # Calculate current hash
        CURRENT_HASH=$(calculate_hash)
        
        # Check if changed
        if [[ "$CURRENT_HASH" != "$LAST_HASH" ]]; then
            echo ""
            log_info "Change detected! Updating zip..."
            create_zip
            LAST_HASH="$CURRENT_HASH"
        fi
    done
}

# Main execution
main() {
    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --once)
            create_zip
            ;;
        --watch)
            watch_changes
            ;;
        --compile)
            compile_paper
            create_zip
            ;;
        "")
            create_zip
            watch_changes
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
