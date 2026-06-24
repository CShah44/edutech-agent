---
name: excalidraw
description: >
  Creates, generates, and exports Excalidraw diagrams (.excalidraw files) from natural language
  descriptions. Use this skill whenever a user asks to draw, diagram, chart, visualize, or sketch
  anything — including system architecture, flowcharts, ER diagrams, sequence diagrams, network
  topology, org charts, mind maps, timelines, or any other visual structure. Trigger even for
  casual phrasing like "can you draw...", "sketch me a...", "make a diagram of...", "show the
  flow of...", or "I need a visual for...". Always use this skill rather than trying to produce
  diagrams inline — the output is a real downloadable .excalidraw file the user can open directly
  at excalidraw.com or in VS Code.
---

# Excalidraw Diagram Skill

## What this skill does

Generates a valid `.excalidraw` JSON file from the user's description and delivers it as a
downloadable file. The user can drag it into [excalidraw.com](https://excalidraw.com) or open it
in the VS Code Excalidraw extension.

---

## Workflow (follow in order)

### Step 1 — Clarify (only if needed)

If the request is ambiguous, ask **one** focused question before proceeding. Usually you can infer
enough to start. Good defaults:
- Diagram type unclear → infer from context (process → flowchart, services → architecture, tables → ER)
- Style not specified → use `roughness: 0`, `fontFamily: 2` (clean/professional)
- Colors not specified → use the semantic palette in `references/schema.md`

Do **not** ask for information you can reasonably assume.

### Step 2 — Plan the diagram

Before writing JSON, think through:

1. **Diagram type**: flowchart / architecture / ERD / sequence / network / org chart / mind map / other
2. **Nodes**: List every box/shape needed with its label
3. **Edges**: List every connection with direction and optional label
4. **Layout**: Choose a layout pattern from `references/layout-patterns.md`
5. **ID scheme**: Assign short, stable IDs now (e.g. `node-1`, `svc-auth`, `db-users`)

Write this plan as a brief internal outline — do **not** show it to the user unless asked.

### Step 3 — Generate the JSON

Read `references/schema.md` for the full element specification before writing JSON.

Key rules:
- Every element needs: `type`, `id`, `x`, `y`, `width`, `height`, plus all required styling fields
- Text labels inside shapes: use a separate `text` element overlaid on the shape (center it)
- Arrow bindings: use `startBinding` / `endBinding` with the target element's `id` — never guess coordinates for arrow endpoints when shapes exist
- IDs must be unique strings across all elements
- `seed` and `versionNonce`: use any integer (e.g. `1`, `2`, `3`… incrementing); they only affect rough.js rendering variation

### Step 4 — Write the file

Write the complete JSON to `/mnt/user-data/outputs/<descriptive-name>.excalidraw`.

Filename should reflect the diagram content: `auth-flow.excalidraw`, `user-service-erd.excalidraw`, etc.

### Step 5 — Deliver

Call `present_files` with the output path. Add a one-sentence summary of what was created and a note that the user can open it at excalidraw.com or in VS Code.

---

## Quality checklist (verify before writing the file)

- [ ] All node IDs referenced in arrows actually exist as elements
- [ ] No two elements share the same `id`
- [ ] Every shape has a corresponding text label (unless intentionally unlabeled)
- [ ] Arrow `startBinding.elementId` and `endBinding.elementId` match real shape IDs
- [ ] Shapes do not overlap (check x/y/width/height against layout)
- [ ] `boundElements` on shapes lists the IDs of arrows connected to them
- [ ] File envelope has `type: "excalidraw"`, `version: 2`, `source`, `elements`, `appState`, `files: {}`

---

## Style defaults

| Property | Default value | Notes |
|---|---|---|
| `roughness` | `0` | Use `1` only when user asks for "sketchy" or "hand-drawn" |
| `fontFamily` | `2` | Helvetica — clean. Use `1` (Virgil) for casual/hand-drawn only |
| `fontSize` | `16` | Node labels. Use `20` for titles, `14` for secondary labels |
| `strokeColor` | `#1e1e1e` | Near-black — works on white background |
| `strokeWidth` | `2` | |
| `strokeStyle` | `"solid"` | Use `"dashed"` for async/optional flows |
| `fillStyle` | `"solid"` | |
| `opacity` | `100` | |
| `angle` | `0` | |

---

## Reference files

- `references/schema.md` — Complete element field reference + worked examples for every element type
- `references/layout-patterns.md` — Coordinate formulas for flowchart, architecture, ERD, sequence, org chart, mind map layouts

Read the relevant reference files **before** generating JSON for any non-trivial diagram.

---

## Common mistakes to avoid

1. **Forgetting text elements** — shapes render without labels if you don't add a separate `text` element (or use a container with `label`)
2. **Arrow coordinate guessing** — always use `startBinding`/`endBinding` instead of hardcoding arrow endpoint coordinates
3. **ID collisions** — if you have a shape `id: "box-1"` and a text `id: "box-1"`, the file will break
4. **Missing `boundElements`** — shapes need `"boundElements": [{"id": "<arrow-id>", "type": "arrow"}]` for arrows to visually attach
5. **Zero-size elements** — every shape needs non-zero `width` and `height`
6. **Overlapping nodes** — always add spacing per the layout patterns; never place two shapes at the same (x, y)