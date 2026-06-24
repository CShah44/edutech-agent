# Fix Pipeline Diagram Overflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the pipeline flowchart from horizontal to vertical layout so it fits within page margins.

**Architecture:** Single file edit to `paper/figures/pipeline.tex` — replace `right of=` positioning with `below=of` vertical flow using TikZ `positioning` library.

**Tech Stack:** LaTeX, TikZ

## Global Constraints

- `main.tex` already loads `\usetikzlibrary{shapes,arrows,positioning}`
- Keep all existing text content, labels, and styling unchanged
- Diagram must fit within `\textwidth`

---

### Task 1: Rewrite pipeline.tex as vertical flow

**Files:**
- Modify: `paper/figures/pipeline.tex`

**Interfaces:**
- Consumes: None (standalone figure file)
- Produces: Updated `pipeline.tex` with vertical layout

- [ ] **Step 1: Rewrite pipeline.tex with vertical layout**

Replace entire file content with:

```latex
% Multi-Agent Pipeline Diagram
% Add to main.tex after \begin{document}

\begin{figure}[t]
\centering
\begin{tikzpicture}[
    stage/.style={rectangle, draw, fill=blue!20, text width=5cm, text centered, rounded corners, minimum height=1cm},
    arrow/.style={->, >=stealth, thick},
    node distance=1.5cm
]

% Nodes - vertical flow
\node[stage] (input) {Questions};
\node[stage, below=of input] (breakdown) {Stage 1: Breakdown\\Question Decomposition};
\node[stage, below=of breakdown] (analysis) {Stage 2: Parallel Analysis\\RAG + Reasoning};
\node[stage, below=of analysis] (synthesis) {Stage 3: Synthesis\\Adaptive Strategy};
\node[stage, below=of synthesis] (creative) {Stage 4: Creative\\ELI5 Transformation};
\node[stage, below=of creative] (output) {ELI5 Answers};

% Arrows
\draw[arrow] (input) -- (breakdown);
\draw[arrow] (breakdown) -- (analysis);
\draw[arrow] (analysis) -- (synthesis);
\draw[arrow] (synthesis) -- (creative);
\draw[arrow] (creative) -- (output);

% Batching label
\node[below=0.5cm of output, text width=8cm, text centered] (batch) {\textbf{Staged Batching:} All questions processed together at each stage\\$\rightarrow$ 100$\times$ reduction in vLLM calls};

\end{tikzpicture}
\caption{Multi-agent architecture for ELI5 explanation generation. Four specialized stages process batches of questions together, enabling staged batching that reduces inference calls by 100$\times$.}
\label{fig:pipeline}
\end{figure}
```

- [ ] **Step 2: Compile and verify**

Run from `paper/` directory:
```bash
cd paper && pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -20
```
Expected: No errors. Diagram renders vertically within page margins.

- [ ] **Step 3: Commit**

```bash
git add paper/figures/pipeline.tex
git commit -m "fix: convert pipeline diagram to vertical layout to prevent page overflow"
```
