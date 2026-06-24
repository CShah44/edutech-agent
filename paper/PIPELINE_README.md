# Paper Pipeline - Overleaf Upload Automation

## Overview

Automated pipeline to create `paper.zip` for Overleaf upload whenever you make changes to the paper.

## Quick Start

### Option 1: Shell Script (Recommended)

```bash
# Create zip and watch for changes
./paper_pipeline.sh

# Or create zip once
./paper_pipeline.sh --once

# Or compile paper first, then create zip
./paper_pipeline.sh --compile
```

### Option 2: Python Script

```bash
# Create zip and watch for changes
./watch_paper.py

# Or create zip once
./watch_paper.py --once
```

### Option 3: Simple Zip Creation

```bash
# Just create the zip file
./create_zip.sh
```

## How It Works

### 1. File Watching
The pipeline watches for changes in these file types:
- `*.tex` (LaTeX files)
- `*.bib` (Bibliography files)
- `*.sty` (Style files)

### 2. Hash Calculation
When a change is detected, the pipeline:
1. Calculates MD5 hash of all paper files
2. Compares with previous hash
3. Only recreates zip if hash changed

### 3. Zip Creation
The zip includes:
- `main.tex` - Main document
- `sections/` - All section files
- `figures/` - All figure files
- `references.bib` - Bibliography
- `compile.sh` - Compilation script

The zip excludes:
- `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out` (auxiliary files)
- `*.zip` (the zip itself)

## File Structure

```
paper/
├── main.tex              # Main document
├── sections/             # Section files
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── figures/              # Figure files
│   ├── pipeline.tex
│   ├── accuracy_quality_tradeoff.tex
│   ├── word_count_comparison.tex
│   ├── correlation_plot.tex
│   └── ttr_comparison.tex
├── references.bib        # Bibliography
├── compile.sh            # Compilation script
├── paper.zip             # Generated zip (for Overleaf)
├── paper_pipeline.sh     # Main pipeline script
├── create_zip.sh         # Zip creation script
├── watch_paper.py        # Python watcher script
└── PIPELINE_README.md    # This file
```

## Usage Examples

### Example 1: Quick Zip Creation
```bash
cd paper
./create_zip.sh
# Upload paper.zip to Overleaf
```

### Example 2: Continuous Development
```bash
cd paper
./paper_pipeline.sh
# Edit files in another terminal
# Zip updates automatically
# Upload to Overleaf when ready
```

### Example 3: Compile and Package
```bash
cd paper
./paper_pipeline.sh --compile
# Paper compiled and zip created
# Upload to Overleaf
```

## Overleaf Upload Process

1. **Run the pipeline**:
   ```bash
   ./paper_pipeline.sh --once
   ```

2. **Go to Overleaf**:
   - Visit https://www.overleaf.com/project
   - Click "New Project" or open existing project

3. **Upload the zip**:
   - Click "Upload" button
   - Select `paper.zip`
   - Overleaf will extract all files

4. **Verify compilation**:
   - Overleaf should auto-compile
   - Check for any errors
   - All figures should render correctly

## Troubleshooting

### Issue: "Permission denied"
**Solution**: Make scripts executable
```bash
chmod +x paper_pipeline.sh create_zip.sh watch_paper.py
```

### Issue: "fswatch not found"
**Solution**: Install fswatch or use Python script
```bash
# Option 1: Install fswatch
brew install fswatch

# Option 2: Use Python script
./watch_paper.py
```

### Issue: "Zip file too large"
**Solution**: Check for unnecessary files
```bash
# Clean auxiliary files
rm -f *.aux *.log *.bbl *.blg *.out

# Recreate zip
./create_zip.sh
```

### Issue: "Figures not showing in Overleaf"
**Solution**: Ensure all figure files are in the figures directory
```bash
ls figures/
# Should show: pipeline.tex, accuracy_quality_tradeoff.tex, etc.
```

## Advanced Features

### Custom Exclusions
Edit `create_zip.sh` to exclude additional files:
```bash
zip -r "$SCRIPT_DIR/$ZIP_NAME" . -x "*.aux" "*.log" "*.bbl" "*.blg" "*.out" "*.your_extension"
```

### Background Mode
Run the watcher in the background:
```bash
# Start background watcher
nohup ./paper_pipeline.sh > paper_watcher.log 2>&1 &

# Check if running
ps aux | grep paper_pipeline

# Stop background watcher
pkill -f paper_pipeline
```

### Git Integration
Add to `.gitignore` to avoid committing the zip:
```
paper.zip
*.aux
*.log
*.bbl
*.blg
*.out
```

## Benefits

1. **Automation**: No manual zip creation needed
2. **Efficiency**: Only recreates when files change
3. **Clean**: Excludes auxiliary files automatically
4. **Portable**: Works on macOS, Linux, Windows (with WSL)
5. **Flexible**: Multiple options for different workflows

## Status

✅ Pipeline is ready to use!

**Next steps**:
1. Run `./paper_pipeline.sh --once` to create initial zip
2. Upload to Overleaf
3. Continue editing with auto-zip updates
