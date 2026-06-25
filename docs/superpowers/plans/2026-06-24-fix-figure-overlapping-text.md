# Plan: Fix Overlapping Text in Figures 1, 2, and 4

## Context
The paper's figures have overlapping text that makes them difficult to read. The issues are in:
1. **Figure 1 (pipeline.tex)**: Annotations on the right side overlap with the diagram nodes and arrow labels
2. **Figure 2 (accuracy_quality_tradeoff.tex)**: All labels use `above right` positioning, causing overlaps when data points are clustered
3. **Figure 4 (correlation_plot.tex)**: Similar label positioning issue plus an annotation box that may overlap with the regression line

## Approach: Minimal Targeted Fixes

Rather than restructuring the entire figures, I'll make surgical changes to fix the overlapping text while preserving the original design and layout.

## Task 1: Fix Figure 1 (pipeline.tex) - Annotations Overlap

**Problem:** Lines 40-41 have annotations positioned at `right=0.5cm of breakdown` and `right=0.5cm of synthesis`. These overlap with:
- Arrow labels ("Search Queries", "Reasoning Points")
- Other diagram elements
- The diagram is already quite dense with nodes and arrows

**Solution:** Remove the overlapping annotations entirely. The diagram is self-explanatory with the arrow labels already describing what each agent does. The caption provides additional context.

**Changes:**
- Delete lines 40-41 (the two annotation nodes)
- Keep the arrow labels which already describe the data flow

**Expected Result:** Cleaner diagram without redundant annotations that were overlapping with other elements.

---

## Task 2: Fix Figure 2 (accuracy_quality_tradeoff.tex) - Clustered Labels

**Problem:** All 7 labels use `above right` positioning (lines 31-37). When data points are close together, labels overlap:
- (26.7, 55.5) Qwen 3B and (26.7, 34.2) Gemma 2B - same x-coordinate
- (2.7, 40.6) Average and (6.3, 40.6) LLaMA 1B - similar y-coordinates  
- (-3.8, 34.2) Mistral 7B and (26.7, 34.2) Gemma 2B - same y-coordinate

**Solution:** Use varied positioning for labels to avoid overlaps:
- Points with high x-values (26.7): Use `right` positioning
- Points with low x-values (-3.8): Use `left` positioning  
- Points with similar y-values: Use different vertical positions (above/below)

**Changes:**
- Line 31: LLaMA 1B (6.3, 40.6) → `below left`
- Line 32: Qwen 7B (2.4, 57.8) → `above`
- Line 33: Mistral 7B (-3.8, 34.2) → `below left`
- Line 34: Qwen 3B (26.7, 55.5) → `above right`
- Line 35: Gemma 2B (26.7, 34.2) → `right`
- Line 36: Gemma 7B (25.0, 6.0) → `below right`
- Line 37: Average (2.7, 40.6) → `below right`

**Expected Result:** No overlapping labels while maintaining readability.

---

## Task 3: Fix Figure 4 (correlation_plot.tex) - Clustered Labels + Annotation

**Problem:** 
1. All 7 labels use `above right` positioning (lines 35-41), causing overlaps in the 90-110 word count cluster
2. The annotation box at (350, 0.23) may overlap with the regression line

**Solution:**
1. Use varied positioning for labels based on their position in the cluster
2. Move the annotation box to the upper-left area where there are no data points

**Changes:**
- Line 35: LLaMA 3B (150.9, 0.2281) → `above left` (isolated point)
- Line 36: Qwen 7B (100.4, 0.2401) → `below right`
- Line 37: Mistral 7B (109.8, 0.2348) → `above right`
- Line 38: Qwen 3B (98.9, 0.2374) → `below left`
- Line 39: LLaMA 1B (92.6, 0.2310) → `below left`
- Line 40: Gemma 2B (105.0, 0.2283) → `above right`
- Line 41: Gemma 7B (100.3, 0.2419) → `above`
- Line 44: Move annotation box from (350, 0.23) to (200, 0.245) - upper-left area

**Expected Result:** No overlapping labels and annotation box doesn't interfere with data visualization.

---

## Execution Steps

1. **Read current figures** to understand exact positions (already done)
2. **Apply minimal changes** to each figure file
3. **Compile paper** to verify fixes work
4. **Test visual output** by opening PDF
5. **Commit changes** if successful

## Files to Modify

- `paper/figures/pipeline.tex` (delete 2 lines)
- `paper/figures/accuracy_quality_tradeoff.tex` (change 7 label positions)
- `paper/figures/correlation_plot.tex` (change 7 label positions + 1 annotation position)

## Verification

1. Run `./compile.sh` in paper directory
2. Open `main.pdf` and visually inspect:
   - Figure 1: No text overlapping with nodes or arrows
   - Figure 2: All 7 model labels are readable and distinct
   - Figure 4: All 7 model labels are readable and annotation box doesn't overlap with regression line

## Risk Assessment

- **Low risk**: Changes are minimal and targeted
- **No functional changes**: Only positioning of text elements, not data or logic
- **Easy to revert**: If fixes cause new issues, can revert individual commits

## Success Criteria

- [ ] Paper compiles without errors
- [ ] Figure 1 has no overlapping annotations
- [ ] Figure 2 has no overlapping labels
- [ ] Figure 4 has no overlapping labels or annotation box
- [ ] All figures remain readable and professionally formatted
