# Fix Pipeline Diagram Overflow

## Problem

The pipeline flowchart in `paper/figures/pipeline.tex` overflows the page width. It uses horizontal positioning with fixed `xshift=3cm` between 4 stages, each with `text width=2.5cm`. The total width exceeds `\textwidth`.

## Solution

Convert the diagram from horizontal to vertical (top-to-bottom) flow.

### Changes

**File:** `paper/figures/pipeline.tex`

- Use `\usetikzlibrary{positioning}` (already imported in `main.tex`)
- Replace `right of=` with `below=of` for stage placement
- Widen node `text width` to ~5cm since horizontal space is now available
- Place Input left of Stage 1, Output right of Stage 1 (or below Input/above Output)
- Move Staged Batching annotation below all stages
- Keep all existing text content and styling

### Layout

```
         [Input]
            |
    [Stage 1: Breakdown]
            |
    [Stage 2: Parallel Analysis]
            |
    [Stage 3: Synthesis]
            |
    [Stage 4: Creative]
            |
        [Output]

  Staged Batching annotation
```

## Scope

Single file edit. No other figures affected.
