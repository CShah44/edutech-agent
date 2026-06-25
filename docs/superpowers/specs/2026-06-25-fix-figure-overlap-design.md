# Paper Figure Cleanup Design

**Date**: 2026-06-25
**Goal**: Remove overlapping text in Figures 1, 2, and 4 while introducing consistent figure styling rules across the paper for future edits.
**Scope**: LaTeX sources under `paper/`, especially shared figure styling in `main.tex` and layout updates in `paper/figures/*.tex`.

## Problem Summary

The paper compiles successfully, but three figures have readability issues:

1. Figure 1 (`paper/figures/pipeline.tex`) has annotation text placed too close to the diagram nodes.
2. Figure 2 (`paper/figures/accuracy_quality_tradeoff.tex`) uses repeated label anchors that crowd nearby points.
3. Figure 4 (`paper/figures/correlation_plot.tex`) places multiple labels and a callout box in a dense region of the plot.

These issues are visual, not data-related. The underlying coordinates and metrics should remain unchanged.

## Design Goals

1. Remove all visible text collisions in Figures 1, 2, and 4.
2. Introduce reusable figure styling so future edits do not reintroduce the same layout problems.
3. Keep the look and structure of the remaining figures consistent with the fixed figures.
4. Preserve the paper’s content and numerical values exactly.

## Non-Goals

1. Do not change any metrics, captions, or figure semantics.
2. Do not rebuild figures as external assets.
3. Do not redesign the paper’s overall visual theme.

## Approach

The cleanup will use a shared styling layer plus small per-figure adjustments.

### Shared Styling Layer

Add reusable TikZ/PGFPlots styles in `paper/main.tex` so all figure files can share the same conventions:

1. A common axis style for chart figures.
2. A common annotation box style for callouts and summary notes.
3. A small set of reusable label-anchor styles for point labels.
4. Consistent font sizing, legend placement, and axis padding defaults.

This keeps the fixes centralized and reduces the chance of future collisions when new data points or labels are added.

### Figure-Specific Layout Fixes

1. Figure 1: move the explanatory annotations into reserved space outside the node flow, using the shared annotation style.
2. Figure 2: vary point label anchors and offsets so each label avoids the nearest point cluster.
3. Figure 4: distribute labels more evenly, move the correlation callout into a lower-density area, and ensure the annotation stays within the axis bounds.
4. Figures 3 and 5: align the bar charts with the shared chart style so typography, spacing, and legends are consistent across the paper.

## Detailed Design

### 1. Shared Figure Styles in `paper/main.tex`

Define a small style block near the existing package setup that can be reused by all figure files:

1. `paperAxis`: standard width, height, grid style, tick label size, legend font, and plot clipping behavior.
2. `paperNote`: white background, rounded corners, inner padding, and a bounded text width for callouts.
3. `paperLabelNE`, `paperLabelNW`, `paperLabelSE`, `paperLabelSW`: small anchor presets for point labels.

The style definitions should be minimal and generic. They should not encode figure-specific coordinates or content.

### 2. Figure 1: Pipeline Diagram

`paper/figures/pipeline.tex` should keep the pipeline structure intact but move the two explanatory notes away from the main node stack.

1. Place annotations in dedicated whitespace below the diagram or in a side area that does not intersect the arrows.
2. Apply the shared note style so the annotations visually match the rest of the paper.
3. If needed, slightly widen the figure canvas or adjust node spacing rather than shrinking text.

The intent is to preserve the diagram’s meaning while removing the visual crowding that caused the overlap.

### 3. Figure 2: Accuracy-Quality Tradeoff

`paper/figures/accuracy_quality_tradeoff.tex` should keep the same plotted coordinates but use a mixed label strategy.

1. Assign label anchors based on local neighborhood rather than using the same anchor for every point.
2. Use small, consistent offsets so the text sits just outside the point cluster.
3. Expand axis padding slightly where needed so labels do not clip against the chart edges.

This figure should still read as a simple scatter plot, but with labels that are distributed intentionally instead of uniformly.

### 4. Figure 4: Correlation Plot

`paper/figures/correlation_plot.tex` needs the most careful label placement because it has a dense cluster near the lower-x region.

1. Use varied anchors so the labels spread around the points instead of stacking in one direction.
2. Move the Pearson correlation callout into an open area with enough white space.
3. Keep the callout inside the axis area and away from the regression line and the main cluster of points.

The goal is to make the regression relationship obvious without the annotations competing with the data points.

### 5. Figures 3 and 5: Style Consistency

`paper/figures/word_count_comparison.tex` and `paper/figures/ttr_comparison.tex` do not need semantic changes, but they should adopt the same shared chart style.

1. Reuse the shared axis defaults so bar width, axis fonts, and legends match the rest of the paper.
2. Keep their existing data and labels unchanged.
3. Make only the minimum visual adjustments needed to align them with the shared style layer.

These figures act as a consistency check: after the cleanup, the paper should look like one coherent visual system rather than a collection of unrelated plots.

## Validation Plan

1. Recompile `paper/main.tex` and confirm the document builds cleanly.
2. Inspect `paper/main.pdf` to verify that Figures 1, 2, and 4 no longer have overlapping text.
3. Confirm that Figures 3 and 5 still match the updated figure style and have not regressed.
4. If any layout issue remains, prefer a local figure-specific adjustment over changing the shared defaults.

## Risks and Mitigations

1. Risk: shared styles may change the appearance of a figure that was already working.
   Mitigation: keep the shared defaults conservative and override locally only where necessary.
2. Risk: label relocation could hide a point or make a callout feel disconnected.
   Mitigation: keep labels close to their data and test in the rendered PDF, not just in source code.
3. Risk: the pipeline figure may become cramped if annotations are moved without enough whitespace.
   Mitigation: reserve explicit space for the annotations instead of placing them opportunistically.

## Acceptance Criteria

1. Figures 1, 2, and 4 render without overlapping text in the compiled PDF.
2. The figure sources share consistent style primitives instead of repeating ad hoc formatting.
3. Figures 3 and 5 remain visually consistent with the rest of the paper.
4. The paper content, captions, and numerical values remain unchanged.