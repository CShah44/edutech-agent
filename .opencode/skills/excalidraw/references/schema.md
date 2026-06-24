# Excalidraw Schema Reference

## Table of Contents
- [Excalidraw Schema Reference](#excalidraw-schema-reference)
  - [Table of Contents](#table-of-contents)
  - [1. File Envelope](#1-file-envelope)
  - [2. Base Element Fields](#2-base-element-fields)
  - [3. Shape Elements](#3-shape-elements)
  - [4. Text Element](#4-text-element)
  - [5. Arrow Element](#5-arrow-element)
  - [6. Line Element](#6-line-element)
  - [7. Frame Element](#7-frame-element)
  - [8. Arrow Binding System](#8-arrow-binding-system)
    - [On the arrow](#on-the-arrow)
    - [On the shape](#on-the-shape)
    - [Connecting arrows (complete pair example)](#connecting-arrows-complete-pair-example)
  - [9. Color Palette](#9-color-palette)
  - [10. Worked Examples](#10-worked-examples)
    - [Example A — 3-step linear flowchart](#example-a--3-step-linear-flowchart)
    - [Example B — Two services with a database (architecture)](#example-b--two-services-with-a-database-architecture)

---

## 1. File Envelope

Every `.excalidraw` file must have this top-level structure:

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://excalidraw.com",
  "elements": [ /* ...element objects... */ ],
  "appState": {
    "viewBackgroundColor": "#ffffff",
    "gridSize": null
  },
  "files": {}
}
```

---

## 2. Base Element Fields

Every element type shares these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✅ | Unique identifier. Use short descriptive strings: `"rect-1"`, `"svc-auth"`, `"arrow-a1"` |
| `type` | string | ✅ | `"rectangle"` \| `"ellipse"` \| `"diamond"` \| `"text"` \| `"arrow"` \| `"line"` \| `"frame"` |
| `x` | number | ✅ | Left edge X coordinate (pixels, origin top-left) |
| `y` | number | ✅ | Top edge Y coordinate (pixels, origin top-left) |
| `width` | number | ✅ | Element width in pixels |
| `height` | number | ✅ | Element height in pixels |
| `angle` | number | ✅ | Rotation in radians. Use `0` for no rotation |
| `strokeColor` | string | ✅ | Hex color for border/stroke. Default: `"#1e1e1e"` |
| `backgroundColor` | string | ✅ | Hex fill color. Use `"transparent"` for no fill |
| `fillStyle` | string | ✅ | `"solid"` \| `"hachure"` \| `"cross-hatch"` \| `"dots"` |
| `strokeWidth` | number | ✅ | Border thickness. Default: `2` |
| `strokeStyle` | string | ✅ | `"solid"` \| `"dashed"` \| `"dotted"` |
| `roughness` | number | ✅ | `0` = clean, `1` = hand-drawn, `2` = very sketchy |
| `opacity` | number | ✅ | `0`–`100`. Default: `100` |
| `groupIds` | array | ✅ | Array of group ID strings. Use `[]` if not grouped |
| `boundElements` | array\|null | ✅ | List of arrows bound to this shape. See §8. Use `null` or `[]` if none |
| `isDeleted` | boolean | ✅ | Always `false` for new elements |
| `locked` | boolean | ✅ | Always `false` unless intentionally locked |
| `link` | string\|null | ✅ | URL hyperlink or `null` |
| `seed` | number | ✅ | Any integer. Affects rough.js rendering variation. Use `1`, `2`, `3`… |
| `version` | number | ✅ | Element version. Use `1` for new elements |
| `versionNonce` | number | ✅ | Any integer. Use same value as `seed` or `1` |
| `updated` | number | ✅ | Unix timestamp ms. Use `1` (static value is fine) |
| `frameId` | string\|null | ✅ | ID of containing frame, or `null` |

---

## 3. Shape Elements

Applies to: `rectangle`, `ellipse`, `diamond`

Additional fields beyond base:

| Field | Type | Required | Description |
|---|---|---|---|
| `roundness` | object\|null | ✅ | Corner rounding. `{"type": 3}` = rounded corners, `null` = sharp |

**Minimal rectangle:**
```json
{
  "id": "rect-1",
  "type": "rectangle",
  "x": 100, "y": 100,
  "width": 200, "height": 80,
  "angle": 0,
  "strokeColor": "#1e1e1e",
  "backgroundColor": "#a5d8ff",
  "fillStyle": "solid",
  "strokeWidth": 2,
  "strokeStyle": "solid",
  "roughness": 0,
  "opacity": 100,
  "groupIds": [],
  "boundElements": [],
  "isDeleted": false,
  "locked": false,
  "link": null,
  "seed": 1, "version": 1, "versionNonce": 1, "updated": 1,
  "frameId": null,
  "roundness": {"type": 3}
}
```

**Shape sizing guidelines:**
- Standard node box: `width: 160–220`, `height: 60–80`
- Database/cylinder: `width: 140`, `height: 80`, use ellipse
- Decision diamond: `width: 160`, `height: 100`
- Large container/swimlane: `width: 400–800`, `height: 200–400`

---

## 4. Text Element

Text elements are used both as standalone labels and as labels overlaid inside shapes.

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | The label content |
| `fontSize` | number | ✅ | Font size in px. Use `16` for body, `20` for titles, `14` for secondary |
| `fontFamily` | number | ✅ | `1` = Virgil (hand-drawn), `2` = Helvetica (clean), `3` = Cascadia (monospace/code) |
| `textAlign` | string | ✅ | `"center"` \| `"left"` \| `"right"` |
| `verticalAlign` | string | ✅ | `"middle"` \| `"top"` \| `"bottom"` |
| `containerId` | string\|null | ✅ | ID of the shape this text is bound inside, or `null` |
| `originalText` | string | ✅ | Same as `text` (used for line-wrap tracking) |
| `lineHeight` | number | ✅ | Use `1.25` |
| `roundness` | null | ✅ | Always `null` for text |
| `baseline` | number | optional | Pixel offset for baseline. Omit or use `0` |

**Positioning text inside a shape:**

Center a text element over its parent shape:
```
text.x = shape.x + (shape.width - text.width) / 2
text.y = shape.y + (shape.height - text.height) / 2
```

For a shape at `x:100, y:100, width:200, height:80` with text width ~120 and height ~20:
```
text.x = 100 + (200 - 120) / 2 = 140
text.y = 100 + (80 - 20) / 2   = 130
```

Estimate text width: ~9px per character at fontSize 16, ~11px per char at fontSize 20.

**Minimal text element:**
```json
{
  "id": "text-1",
  "type": "text",
  "x": 140, "y": 130,
  "width": 120, "height": 20,
  "angle": 0,
  "strokeColor": "#1e1e1e",
  "backgroundColor": "transparent",
  "fillStyle": "solid",
  "strokeWidth": 1,
  "strokeStyle": "solid",
  "roughness": 0,
  "opacity": 100,
  "groupIds": [],
  "boundElements": [],
  "isDeleted": false,
  "locked": false,
  "link": null,
  "seed": 2, "version": 1, "versionNonce": 1, "updated": 1,
  "frameId": null,
  "roundness": null,
  "text": "My Label",
  "fontSize": 16,
  "fontFamily": 2,
  "textAlign": "center",
  "verticalAlign": "middle",
  "containerId": null,
  "originalText": "My Label",
  "lineHeight": 1.25
}
```

---

## 5. Arrow Element

Arrows connect shapes. Always use `startBinding`/`endBinding` — do not manually compute endpoint coordinates when source/target shapes exist.

| Field | Type | Required | Description |
|---|---|---|---|
| `points` | array | ✅ | Array of `[x, y]` pairs relative to arrow's own `x,y`. Minimum: `[[0,0],[dx,dy]]` |
| `startBinding` | object\|null | ✅ | Binding to source shape. See §8 |
| `endBinding` | object\|null | ✅ | Binding to target shape. See §8 |
| `startArrowhead` | string\|null | ✅ | `null` \| `"arrow"` \| `"triangle"` \| `"dot"` \| `"bar"` |
| `endArrowhead` | string\|null | ✅ | `null` \| `"arrow"` \| `"triangle"` \| `"dot"` \| `"bar"` |
| `elbowed` | boolean | optional | `true` = right-angle routing (elbow arrows) |
| `roundness` | object\|null | ✅ | `{"type": 2}` for curved arrows, `null` for straight |

**Standard directional arrow (A → B):**
```json
{
  "id": "arrow-1",
  "type": "arrow",
  "x": 300, "y": 140,
  "width": 100, "height": 1,
  "angle": 0,
  "strokeColor": "#1e1e1e",
  "backgroundColor": "transparent",
  "fillStyle": "solid",
  "strokeWidth": 2,
  "strokeStyle": "solid",
  "roughness": 0,
  "opacity": 100,
  "groupIds": [],
  "boundElements": [],
  "isDeleted": false,
  "locked": false,
  "link": null,
  "seed": 10, "version": 1, "versionNonce": 1, "updated": 1,
  "frameId": null,
  "roundness": {"type": 2},
  "points": [[0, 0], [100, 0]],
  "startBinding": {
    "elementId": "rect-source",
    "focus": 0,
    "gap": 8
  },
  "endBinding": {
    "elementId": "rect-target",
    "focus": 0,
    "gap": 8
  },
  "startArrowhead": null,
  "endArrowhead": "arrow"
}
```

**Arrow point calculation:**

When drawing a horizontal arrow from shape A (right edge) to shape B (left edge):
```
arrow.x = A.x + A.width          (right edge of A)
arrow.y = A.y + A.height / 2     (vertical midpoint of A)
arrow.points = [[0, 0], [gap, 0]] where gap = B.x - (A.x + A.width)
```

For vertical arrow from A (bottom) to B (top):
```
arrow.x = A.x + A.width / 2
arrow.y = A.y + A.height
arrow.points = [[0, 0], [0, gap]] where gap = B.y - (A.y + A.height)
```

**Arrow label:** Add a separate `text` element positioned at the midpoint of the arrow. Set `containerId: null`. No binding needed.

---

## 6. Line Element

Lines do not connect shapes — they are decorative or structural separators.

| Field | Type | Required | Description |
|---|---|---|---|
| `points` | array | ✅ | Array of `[x, y]` pairs relative to element's own `x,y`. Start at `[0, 0]` |
| `startArrowhead` | null | ✅ | Always `null` for lines |
| `endArrowhead` | null | ✅ | Always `null` for lines |
| `roundness` | object\|null | ✅ | `{"type": 2}` for curved, `null` for straight |

---

## 7. Frame Element

Frames group elements visually with a labeled border.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Frame title displayed at top |
| `roundness` | null | ✅ | Always `null` |

Elements inside a frame set `frameId` to the frame's `id`.

---

## 8. Arrow Binding System

This is the most important part to get right. Binding connects arrows to shapes so Excalidraw can re-route them when shapes move.

### On the arrow
```json
"startBinding": {
  "elementId": "the-source-shape-id",
  "focus": 0,
  "gap": 8
},
"endBinding": {
  "elementId": "the-target-shape-id",
  "focus": 0,
  "gap": 8
}
```

- `elementId`: The `id` of the shape being connected
- `focus`: `-1` to `1`. `0` = center of the edge, `-1` = top/left, `1` = bottom/right
- `gap`: Pixels between arrow endpoint and shape border. Use `8` as default

### On the shape
Every shape that has arrows connected to it must list them in `boundElements`:
```json
"boundElements": [
  {"id": "arrow-1", "type": "arrow"},
  {"id": "arrow-2", "type": "arrow"}
]
```

If a shape has no arrows, use `"boundElements": []` or `null`.

### Connecting arrows (complete pair example)

Shape A with arrow going to Shape B:
```json
// Shape A — has an outgoing arrow
{ "id": "shape-a", ..., "boundElements": [{"id": "arr-1", "type": "arrow"}] }

// Shape B — has an incoming arrow  
{ "id": "shape-b", ..., "boundElements": [{"id": "arr-1", "type": "arrow"}] }

// The arrow
{
  "id": "arr-1", "type": "arrow",
  ...,
  "startBinding": {"elementId": "shape-a", "focus": 0, "gap": 8},
  "endBinding":   {"elementId": "shape-b", "focus": 0, "gap": 8},
  "startArrowhead": null,
  "endArrowhead": "arrow"
}
```

---

## 9. Color Palette

Use these consistently for semantic meaning:

| Purpose | Background | Stroke |
|---|---|---|
| Process / action | `#a5d8ff` (light blue) | `#1971c2` |
| Start / success / data store | `#b2f2bb` (light green) | `#2f9e44` |
| Decision / warning | `#fff3bf` (light yellow) | `#e67700` |
| Error / termination | `#ffc9c9` (light red) | `#c92a2a` |
| External system | `#e9ecef` (light grey) | `#495057` |
| Neutral / default | `transparent` | `#1e1e1e` |
| Highlight / important | `#d3f9d8` (mint green) | `#1e1e1e` |
| Database / storage | `#dbe4ff` (light indigo) | `#3b5bdb` |
| User / actor | `#fff9db` (light cream) | `#e67700` |

**Arrow stroke colors:**
- Synchronous call: `#1e1e1e` solid
- Async / response: `#1e1e1e` dashed (`"strokeStyle": "dashed"`)
- Data flow: `#1971c2` solid
- Error path: `#c92a2a` dashed

---

## 10. Worked Examples

### Example A — 3-step linear flowchart

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://excalidraw.com",
  "appState": {"viewBackgroundColor": "#ffffff", "gridSize": null},
  "files": {},
  "elements": [
    {
      "id": "start", "type": "rectangle",
      "x": 100, "y": 100, "width": 160, "height": 60, "angle": 0,
      "strokeColor": "#2f9e44", "backgroundColor": "#b2f2bb", "fillStyle": "solid",
      "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [{"id": "a1", "type": "arrow"}],
      "isDeleted": false, "locked": false, "link": null,
      "seed": 1, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": {"type": 3}
    },
    {
      "id": "t-start", "type": "text",
      "x": 147, "y": 120, "width": 66, "height": 20, "angle": 0,
      "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
      "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [], "isDeleted": false, "locked": false, "link": null,
      "seed": 2, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": null,
      "text": "Start", "fontSize": 16, "fontFamily": 2,
      "textAlign": "center", "verticalAlign": "middle",
      "containerId": null, "originalText": "Start", "lineHeight": 1.25
    },
    {
      "id": "process", "type": "rectangle",
      "x": 360, "y": 100, "width": 160, "height": 60, "angle": 0,
      "strokeColor": "#1971c2", "backgroundColor": "#a5d8ff", "fillStyle": "solid",
      "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [{"id": "a1", "type": "arrow"}, {"id": "a2", "type": "arrow"}],
      "isDeleted": false, "locked": false, "link": null,
      "seed": 3, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": {"type": 3}
    },
    {
      "id": "t-process", "type": "text",
      "x": 393, "y": 120, "width": 94, "height": 20, "angle": 0,
      "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
      "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [], "isDeleted": false, "locked": false, "link": null,
      "seed": 4, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": null,
      "text": "Process Data", "fontSize": 16, "fontFamily": 2,
      "textAlign": "center", "verticalAlign": "middle",
      "containerId": null, "originalText": "Process Data", "lineHeight": 1.25
    },
    {
      "id": "end", "type": "rectangle",
      "x": 620, "y": 100, "width": 160, "height": 60, "angle": 0,
      "strokeColor": "#c92a2a", "backgroundColor": "#ffc9c9", "fillStyle": "solid",
      "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [{"id": "a2", "type": "arrow"}],
      "isDeleted": false, "locked": false, "link": null,
      "seed": 5, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": {"type": 3}
    },
    {
      "id": "t-end", "type": "text",
      "x": 671, "y": 120, "width": 58, "height": 20, "angle": 0,
      "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
      "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [], "isDeleted": false, "locked": false, "link": null,
      "seed": 6, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": null,
      "text": "End", "fontSize": 16, "fontFamily": 2,
      "textAlign": "center", "verticalAlign": "middle",
      "containerId": null, "originalText": "End", "lineHeight": 1.25
    },
    {
      "id": "a1", "type": "arrow",
      "x": 260, "y": 130, "width": 100, "height": 0, "angle": 0,
      "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
      "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [], "isDeleted": false, "locked": false, "link": null,
      "seed": 10, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": {"type": 2},
      "points": [[0, 0], [100, 0]],
      "startBinding": {"elementId": "start",   "focus": 0, "gap": 8},
      "endBinding":   {"elementId": "process", "focus": 0, "gap": 8},
      "startArrowhead": null, "endArrowhead": "arrow"
    },
    {
      "id": "a2", "type": "arrow",
      "x": 520, "y": 130, "width": 100, "height": 0, "angle": 0,
      "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
      "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
      "groupIds": [], "boundElements": [], "isDeleted": false, "locked": false, "link": null,
      "seed": 11, "version": 1, "versionNonce": 1, "updated": 1, "frameId": null,
      "roundness": {"type": 2},
      "points": [[0, 0], [100, 0]],
      "startBinding": {"elementId": "process", "focus": 0, "gap": 8},
      "endBinding":   {"elementId": "end",     "focus": 0, "gap": 8},
      "startArrowhead": null, "endArrowhead": "arrow"
    }
  ]
}
```

### Example B — Two services with a database (architecture)

Pattern: left-to-right services, database below center service.

```
[Client] ──► [API Service] ──► [Auth Service]
                   │
                   ▼
              [PostgreSQL DB]
```

Key coordinates:
- Client:       x:80,  y:200, w:160, h:70
- API Service:  x:340, y:200, w:160, h:70
- Auth Service: x:600, y:200, w:160, h:70
- DB:           x:340, y:380, w:160, h:70

Arrows: client→api (horizontal), api→auth (horizontal), api→db (vertical downward).