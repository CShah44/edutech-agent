# LaTeX Paper Writing Skill

## Compilation
- Always produce Overleaf-compatible LaTeX (pdflatex or xelatex)
- Use \documentclass{IEEEtran} or similar — ask user for venue
- Split into section files with \input{}
- Use natbib or biblatex for references

## Quality Checks
- Every claim must trace back to code or data in the project
- Add \label{} and \ref{} for all figures and tables
- Include an abstract under 250 words
- Add \usepackage{} declarations at the top of main.tex only