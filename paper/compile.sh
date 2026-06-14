#!/bin/bash
# Compile the LaTeX paper

echo "Compiling LaTeX paper..."
cd "$(dirname "$0")"

# First pass
echo "Pass 1: pdflatex..."
pdflatex -interaction=nonstopmode main.tex

# Generate bibliography
echo "Pass 2: bibtex..."
bibtex main

# Second pass to incorporate citations
echo "Pass 3: pdflatex..."
pdflatex -interaction=nonstopmode main.tex

# Third pass to finalize references
echo "Pass 4: pdflatex..."
pdflatex -interaction=nonstopmode main.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.bbl *.blg *.out

echo "Done! Check main.pdf"
