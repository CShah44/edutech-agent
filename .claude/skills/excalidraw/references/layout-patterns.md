# Excalidraw Layout Patterns

Coordinate formulas and spacing rules for common diagram types.
All measurements in pixels. Canvas origin (0,0) is top-left.

## Table of Contents
- [Excalidraw Layout Patterns](#excalidraw-layout-patterns)
  - [Table of Contents](#table-of-contents)
  - [1. General Spacing Rules](#1-general-spacing-rules)
  - [2. Flowchart — Left-to-Right](#2-flowchart--left-to-right)
  - [3. Flowchart — Top-to-Bottom](#3-flowchart--top-to-bottom)
  - [4. System Architecture — Layers](#4-system-architecture--layers)
  - [5. System Architecture — Left-to-Right Services](#5-system-architecture--left-to-right-services)
  - [6. Entity-Relationship Diagram (ERD)](#6-entity-relationship-diagram-erd)
  - [7. Sequence Diagram](#7-sequence-diagram)
  - [8. Org Chart — Top-Down Hierarchy](#8-org-chart--top-down-hierarchy)
  - [9. Mind Map — Radial](#9-mind-map--radial)
  - [10. Network Topology](#10-network-topology)
  - [11. Decision Tree](#11-decision-tree)
  - [Choosing a Layout](#choosing-a-layout)

---

## 1. General Spacing Rules

| Relationship | Spacing |
|---|---|
| Between sibling nodes at same level | 60px gap (node edge to node edge) |
| Between levels / layers | 80–100px gap |
| Arrow gap to shape border | 8px (use `gap: 8` in binding) |
| Canvas starting offset | Begin at x:80, y:80 — avoids clipping at edge |
| Standard node size | width:160–200, height:60–70 |
| Wide node (service box) | width:180, height:80 |
| Narrow node (actor) | width:120, height:60 |
| Decision diamond | width:160, height:100 |
| Database ellipse | width:140, height:80 |

**ID convention:** Use descriptive prefixes: `svc-`, `db-`, `node-`, `arrow-`, `text-`, `frame-`, `actor-`

---

## 2. Flowchart — Left-to-Right

**When to use:** Step-by-step process, pipeline stages, user journey.

```
[Step 1] ──► [Step 2] ──► [Step 3] ──► ...
```

**Formula:**
```
NODE_W = 160
NODE_H = 60
H_GAP  = 100   ← horizontal gap between nodes

node[i].x = 80 + i * (NODE_W + H_GAP)
node[i].y = 200  ← fixed vertical center

arrow[i].x     = node[i].x + NODE_W
arrow[i].y     = node[i].y + NODE_H / 2
arrow[i].width = H_GAP
arrow[i].points = [[0, 0], [H_GAP, 0]]
```

**With decision branch (diamond):**
- Diamond is placed at branch point with same formula
- "Yes" branch: horizontal arrow continues right
- "No" branch: vertical arrow goes down 160px, then another node below

Example coordinates for a 4-node flow:
```
Start:    x:80,  y:200
Step A:   x:340, y:200
Decision: x:600, y:190  (h:100 so center at y:240)
Step B:   x:860, y:200
Alt:      x:600, y:380  (below decision — "No" path)
```

---

## 3. Flowchart — Top-to-Bottom

**When to use:** Approval workflows, decision trees, hierarchical processes.

```
       [Start]
          │
          ▼
      [Step A]
          │
          ▼
     [Decision?]
      /        \
   [Yes]      [No]
```

**Formula:**
```
NODE_W = 180
NODE_H = 60
V_GAP  = 80

node[i].x = CANVAS_CENTER_X - NODE_W / 2  ← center main path
node[i].y = 80 + i * (NODE_H + V_GAP)

CANVAS_CENTER_X = 400  ← typical

arrow.x = node[i].x + NODE_W / 2
arrow.y = node[i].y + NODE_H
arrow.points = [[0, 0], [0, V_GAP]]
```

**Branch children (at same level):**
```
left_child.x  = CANVAS_CENTER_X - NODE_W - 80
right_child.x = CANVAS_CENTER_X + 80
both.y        = decision.y + decision.height + V_GAP
```

---

## 4. System Architecture — Layers

**When to use:** Frontend / Backend / Database stack, OSI layers, infra tiers.

```
┌──────────────────────────────────────────┐  ← Frame: "Presentation"
│  [Browser]          [Mobile App]         │
└──────────────────────────────────────────┘
              │               │
              ▼               ▼
┌──────────────────────────────────────────┐  ← Frame: "Application"
│  [API Gateway]    [Auth Service]         │
└──────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐  ← Frame: "Data"
│  [PostgreSQL]   [Redis Cache]            │
└──────────────────────────────────────────┘
```

**Formula:**
```
FRAME_X = 60
FRAME_W = 700
FRAME_H = 120
LAYER_GAP = 60   ← gap between frames

layer[i].frame.x = FRAME_X
layer[i].frame.y = 80 + i * (FRAME_H + LAYER_GAP)
layer[i].frame.width = FRAME_W

nodes inside frame:
  node.y = frame.y + 30
  node.x = frame.x + 40 + j * (NODE_W + 60)
```

Use a `rectangle` with `backgroundColor: "transparent"` and `strokeStyle: "dashed"` as the layer container if not using frames. Place a title text above each layer.

---

## 5. System Architecture — Left-to-Right Services

**When to use:** Microservices, API chains, request flow across services.

```
[Client] ──► [API Gateway] ──► [Service A] ──► [DB A]
                    │
                    └──────► [Service B] ──► [DB B]
```

**Formula:**
```
TIER_X = [80, 300, 540, 780]   ← X positions for each tier
SERVICE_Y_BASE = 200
SERVICE_GAP    = 160            ← vertical gap between services in same tier

service.x = TIER_X[tier]
service.y = SERVICE_Y_BASE + rank_in_tier * (NODE_H + SERVICE_GAP)
```

Databases sit to the right (+240px x) of the service that owns them, at the same Y.

---

## 6. Entity-Relationship Diagram (ERD)

**When to use:** Database schema, data model visualization.

**Shape conventions:**
- Entity table: `rectangle`, width:200, height depends on field count (40 + 30*n)
- Primary key field: bold text (capitalize label), `backgroundColor: "#dbe4ff"`
- Foreign key: italic text label
- Relationship line: `arrow` with `"endArrowhead": null` and `"startArrowhead": null` for one-to-one; use `"endArrowhead": "arrow"` for one-to-many

**Table layout:**
```
TABLE_W    = 200
ROW_H      = 30
HEADER_H   = 40
H_GAP      = 120   ← gap between tables horizontally
V_GAP      = 80    ← gap between tables vertically

For n tables in a grid with max 3 columns:
  col = i % 3
  row = i // 3
  table.x = 80 + col * (TABLE_W + H_GAP)
  table.y = 80 + row * (estimated_table_height + V_GAP)
```

**Table structure (elements):**
1. Header rectangle: full width, height:40, `backgroundColor: "#dbe4ff"`
2. Header text: table name, centered in header
3. Per field: rectangle row, height:30, alternating white / `#f8f9fa`
4. Per field: text element with field name + type

---

## 7. Sequence Diagram

**When to use:** API calls, protocol flows, actor interactions over time.

```
Actor A        Actor B        Actor C
   │               │               │
   │──request──►   │               │
   │               │──query──────► │
   │               │ ◄──result──── │
   │ ◄──response── │               │
   │               │               │
```

**Formula:**
```
ACTOR_X_STEP = 220   ← horizontal distance between actors
ACTOR_Y      = 80
LIFELINE_Y_START = 130
LIFELINE_Y_END   = 80 + message_count * 80 + 60
MSG_Y_STEP   = 80

actor[i].x = 80 + i * ACTOR_X_STEP
actor[i].y = ACTOR_Y

lifeline[i] is a vertical dashed line:
  x = actor[i].x + ACTOR_W / 2
  y = LIFELINE_Y_START
  points = [[0,0], [0, LIFELINE_Y_END - LIFELINE_Y_START]]
  strokeStyle = "dashed"

message[j] is a horizontal arrow:
  y = LIFELINE_Y_START + j * MSG_Y_STEP + 40
  x = from_actor_lifeline_x (or to_actor_lifeline_x if going left)
  points direction: positive dx = going right, negative dx = going left
```

Activation boxes: narrow rectangles (width:12) overlaid on lifelines during active processing.

---

## 8. Org Chart — Top-Down Hierarchy

**When to use:** Company structure, reporting lines, project hierarchy.

```
         [CEO]
        /     \
    [CTO]    [CFO]
    /   \
 [Eng] [QA]
```

**Formula:**
```
NODE_W = 160
NODE_H = 60
V_GAP  = 80
H_GAP  = 40   ← between siblings

For balanced binary tree, level l, position p in level:
  total_width_at_level = count[l] * NODE_W + (count[l]-1) * H_GAP
  node.x = (CANVAS_W - total_width_at_level) / 2 + p * (NODE_W + H_GAP)
  node.y = 80 + l * (NODE_H + V_GAP)
```

For deeper trees, use CANVAS_W = 1200 and increase H_GAP proportionally.

---

## 9. Mind Map — Radial

**When to use:** Brainstorming, topic exploration, concept clustering.

**Structure:** Central topic → branches → sub-branches

```
         [Sub A1]──[A1]─┐
[Sub A2]──[A2]─┘        │
                   [Center]──[B1]──[Sub B1]
         [Sub C1]──[C1]─┐
[Sub C2]──[C2]─┘
```

**Formula (simplified radial placement):**
```
CENTER_X = 500, CENTER_Y = 400
MAIN_RADIUS = 200   ← distance from center to main branches

For n main branches, angle_step = 360 / n:
  branch[i].angle = i * angle_step  (degrees)
  branch[i].x = CENTER_X + MAIN_RADIUS * cos(angle_rad) - NODE_W/2
  branch[i].y = CENTER_Y + MAIN_RADIUS * sin(angle_rad) - NODE_H/2

SUB_RADIUS = 150   ← additional radius from branch to sub-branch
  sub.x = branch.center_x + SUB_RADIUS * cos(same_direction_angle) - NODE_W/2
  sub.y = branch.center_y + SUB_RADIUS * sin(same_direction_angle) - NODE_H/2
```

Simplified hand-calculation for 4 branches:
- Right:  angle=0°,   x=center_x+200, y=center_y
- Bottom: angle=90°,  x=center_x,     y=center_y+200
- Left:   angle=180°, x=center_x-200, y=center_y
- Top:    angle=270°, x=center_x,     y=center_y-200

Use lines (not arrows) to connect branches for mind maps.

---

## 10. Network Topology

**When to use:** Infrastructure diagrams, cloud architecture, network maps.

**Shape conventions:**
- Server/VM: `rectangle`, width:140, height:70
- Database: `ellipse`, width:140, height:80
- Load balancer: `diamond`, width:140, height:100
- Internet/cloud: `ellipse`, width:180, height:100, `strokeStyle: "dashed"`
- Firewall: `rectangle`, `backgroundColor: "#ffc9c9"`, width:140, height:50
- Router/switch: `diamond`, width:100, height:100

**Layout (3-tier typical):**
```
Tier 1 (Internet):  y:80,  single cloud shape, centered
Tier 2 (Edge):      y:260, load balancer + firewall, centered
Tier 3 (App):       y:440, 2–4 app servers spread horizontally
Tier 4 (Data):      y:620, databases + cache, spread horizontally
```

---

## 11. Decision Tree

**When to use:** Business logic branching, troubleshooting guides, yes/no flows.

**Shape conventions:**
- Decision: `diamond`
- Outcome/leaf: `rectangle`, width:160, height:60
- Root question: `rectangle` at top center

**Formula (binary tree, 3 levels):**
```
Level 0 (root):   x:440, y:80
Level 1 (2 nodes): x:220 and x:660, y:280
Level 2 (4 nodes): x:80, x:360, x:540, x:820, y:480
```

Arrow labels ("Yes"/"No") are text elements placed near arrowhead midpoints.

---

## Choosing a Layout

| User says... | Use layout |
|---|---|
| "flow", "process", "steps", "pipeline" | Flowchart LTR or TTB |
| "architecture", "services", "microservices", "system" | Architecture LTR or Layers |
| "database", "schema", "entities", "tables", "ER" | ERD |
| "sequence", "API calls", "protocol", "messages between" | Sequence |
| "org chart", "reporting", "team structure" | Org Chart |
| "mind map", "brainstorm", "topics", "concepts" | Mind Map Radial |
| "network", "infra", "cloud", "servers", "topology" | Network Topology |
| "decision", "yes/no", "branching logic", "troubleshoot" | Decision Tree |