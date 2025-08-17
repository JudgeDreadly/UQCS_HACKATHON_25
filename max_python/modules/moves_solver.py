# modules/moves_solver.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

ENEMY_RADIUS = 0.8
X_GROUP_TOL  = 0.6
X_PRE_EPS    = 0.35
X_POST_EPS   = 0.25
STEP_STEEPNESS = 55.0

MOVE_DELTAS = {"R":(1,0),"UR":(1,-1),"DR":(1,1),"U":(0,-1),"D":(0,1)}

@dataclass
class GridSpec:
    dx_units: float = 0.25      # finer grid by default
    dy_units: float = 0.25
    max_cols: int = 500
    max_rows: int = 400

def step_term(k: float, a: float, x_at: float) -> str:
    return f"{k:.4f}/(1+exp(-{a:.0f}*(x+({-x_at:+.4f}))))"

def diag_params_from_line(xs: float, xe: float, slope_m: float) -> Tuple[float,float,float]:
    start, end = (xs, xe) if xs < xe else (xe, xs)
    a = abs(slope_m) / 2.0
    if slope_m >= 0: b, c = -start, -end
    else:            b, c = -end,   -start
    return a, b, c

def diag_term(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = x2-x1; dy = y2-y1
    if abs(dx)<1e-9 and abs(dy)<1e-9: return "0"
    m  = dy/dx if dx!=0 else 0.0
    a,b,c = diag_params_from_line(x1, x2, m)
    return f"{a:.4f}*(abs(x+({b:+.4f}))-abs(x+({c:+.4f})))"

def moves_to_expression(moves: List[str], start_xy: Tuple[float,float], gs: GridSpec, step_a: float = STEP_STEEPNESS) -> str:
    x,y = start_xy
    parts=[]
    for m in moves:
        if m=="R":
            x += gs.dx_units
        elif m=="U":
            dy = -gs.dy_units; parts.append(step_term(dy, step_a, x)); y += dy
        elif m=="D":
            dy = +gs.dy_units; parts.append(step_term(dy, step_a, x)); y += dy
        elif m=="UR":
            nx,ny = x+gs.dx_units, y-gs.dy_units; parts.append(diag_term(x,y,nx,ny)); x,y=nx,ny
        elif m=="DR":
            nx,ny = x+gs.dx_units, y+gs.dy_units; parts.append(diag_term(x,y,nx,ny)); x,y=nx,ny
    expr = " + ".join([p for p in parts if p!="0"]).replace("+-","-").replace("--","+").strip(" +")
    return expr if expr else "0"

def group_enemies_by_x(enemies_xy: List[Tuple[float,float]], start_x: float, x_tol: float=X_GROUP_TOL):
    cand=[p for p in enemies_xy if p[0]>=start_x]; cand.sort(key=lambda p:p[0])
    cols=[]; cur=[]; cur_x=None
    for x,y in cand:
        if cur_x is None or abs(x-cur_x)<=x_tol:
            cur.append((x,y)); cur_x = x if cur_x is None else (cur_x+x)/2.0
        else:
            cols.append(cur); cur=[(x,y)]; cur_x=x
    if cur: cols.append(cur)
    return cols

def make_occ_grid(board, obs_for_planning: np.ndarray, gs: GridSpec) -> np.ndarray:
    cols = int(np.ceil((board.x_range[1]-board.x_range[0]) / gs.dx_units))
    rows = int(np.ceil((board.y_range[1]-board.y_range[0]) / gs.dy_units))
    cols = min(cols, gs.max_cols); rows = min(rows, gs.max_rows)
    occ = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            x = board.x_range[0] + c*gs.dx_units
            y = board.y_range[0] + r*gs.dy_units  # r=0 => y_min (bottom)
            px,py = board.xy_to_px(x,y)          # board handles y axis inversion
            px = int(np.clip(px,0,board.w-1)); py = int(np.clip(py,0,board.h-1))
            occ[r,c] = 1 if obs_for_planning[py,px] > 0 else 0
    return occ

def xy_to_cell(board, xy: Tuple[float,float], gs: GridSpec, occ: np.ndarray) -> Tuple[int,int]:
    x,y = xy
    c = int(np.clip(np.floor((x - board.x_range[0]) / gs.dx_units), 0, occ.shape[1]-1))
    r = int(np.clip(np.floor((y - board.y_range[0]) / gs.dy_units), 0, occ.shape[0]-1))
    return (c,r)

def cell_to_xy(board, cell: Tuple[int,int], gs: GridSpec) -> Tuple[float,float]:
    c,r = cell
    x = board.x_range[0] + c*gs.dx_units
    y = board.y_range[0] + r*gs.dy_units
    return (x,y)

def plan_cells(occ: np.ndarray, start_cell: Tuple[int,int], goal_cell: Tuple[int,int], algo: str="A*"):
    from modules.pathfinding_algos import PLANNERS
    planner = PLANNERS.get(algo, PLANNERS["A*"])
    return planner(occ, start_cell, goal_cell)  # [(move, cell), ...]

def _draw_critical_grid(
    occ_plan: np.ndarray,
    occ_main: np.ndarray,
    path_cells: List[Tuple[int,int]],
    actor_cells: Dict[str, List[Tuple[int,int]]],
    scale: int = 10
) -> np.ndarray:
    """
    Draw a clean grid image:
      - light green: original obstacles (occ_main)
      - dark green: tolerance band (occ_plan - occ_main)
      - blue dots: allies, red dots: enemies, yellow ring: shooter
      - red polyline: path
    Top of the image is +Y.
    """
    H, W = occ_plan.shape

    # Flip vertically for display so that row 0 (y_min/bottom) is at the bottom.
    disp_main = np.flipud(occ_main)
    disp_plan = np.flipud(occ_plan)

    img = np.full((H*scale, W*scale, 3), 255, np.uint8)

    base_mask = disp_main.repeat(scale, 0).repeat(scale, 1) > 0
    tol_bool  = (disp_plan > 0) & (disp_main == 0)
    tol_mask  = tol_bool.repeat(scale, 0).repeat(scale, 1) > 0

    LIGHT_GREEN = (180, 255, 180)
    DARK_GREEN  = (100, 200, 100)
    RED         = (0, 0, 255)
    BLUE        = (255, 0, 0)
    YELLOW      = (0, 215, 255)
    GRID_LINE   = (230, 230, 230)

    img[base_mask] = LIGHT_GREEN
    img[tol_mask]  = DARK_GREEN

    # faint grid
    for r in range(0, H*scale, scale):
        cv2.line(img, (0, r), (W*scale-1, r), GRID_LINE, 1)
    for c in range(0, W*scale, scale):
        cv2.line(img, (c, 0), (c, H*scale-1), GRID_LINE, 1)

    def disp_pt(cell):
        c, r = cell
        rr = (H - 1 - r)
        return (c * scale + scale // 2, rr * scale + scale // 2)

    pts = [disp_pt(cr) for cr in path_cells]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], RED, 2)

    for cell in actor_cells.get("enemies", []):
        cv2.circle(img, disp_pt(cell), max(2, scale//3), RED, -1)
    for cell in actor_cells.get("allies", []):
        cv2.circle(img, disp_pt(cell), max(2, scale//3), BLUE, -1)
    if actor_cells.get("shooter"):
        cv2.circle(img, disp_pt(actor_cells["shooter"]), max(3, scale//2), YELLOW, 2)

    return img

def solve_moves_on_components(
    *,
    board,
    roi: np.ndarray,
    obs_main: np.ndarray,
    obs_plan: np.ndarray,
    obs_contours,
    actors,
    shooter,
    grid_dx: float,
    grid_dy: float,
    algo: str,
    overlay_path: Optional[str] = None
) -> Dict:
    gs = GridSpec(dx_units=grid_dx, dy_units=grid_dy)
    occ = make_occ_grid(board, obs_plan, gs)  # plan against tolerant mask

    enemies_xy = [(a.x, a.y) for a in actors if a.side == "enemy"]
    path_cells: List[Tuple[int,int]] = []
    path_moves: List[str] = []

    if not enemies_xy:
        start_xy = (shooter.x, shooter.y)
        goal_xy  = (board.x_range[1] - gs.dx_units, shooter.y)
        start_cell = xy_to_cell(board, start_xy, gs, occ)
        goal_cell  = xy_to_cell(board, goal_xy,  gs, occ)
        crumb = plan_cells(occ, start_cell, goal_cell, algo)
        path_cells = [start_cell] + [cell for (_,cell) in crumb]
        path_moves = [m for (m,_) in crumb]
    else:
        cols = group_enemies_by_x(enemies_xy, start_x=shooter.x, x_tol=X_GROUP_TOL)
        cur_xy = (shooter.x, shooter.y)
        cur_cell = xy_to_cell(board, cur_xy, gs, occ)
        path_cells.append(cur_cell)
        for col in cols:
            xcol = float(np.mean([x for (x,_) in col]))
            x_pre  = xcol - 0.35
            x_post = xcol + 0.25
            remaining = col[:]
            while remaining:
                remaining.sort(key=lambda p: abs(p[1] - cur_xy[1]))
                ex, ey = remaining.pop(0)

                goal1_cell = xy_to_cell(board, (x_pre, ey), gs, occ)
                crumb1 = plan_cells(occ, cur_cell, goal1_cell, algo)
                if not crumb1:
                    continue
                path_moves.extend([m for (m,_) in crumb1])
                path_cells.extend([cell for (_,cell) in crumb1])
                cur_cell = path_cells[-1]; cur_xy = cell_to_xy(board, cur_cell, gs)

                goal2_cell = xy_to_cell(board, (x_post, cur_xy[1]), gs, occ)
                crumb2 = plan_cells(occ, cur_cell, goal2_cell, algo)
                if crumb2:
                    path_moves.extend([m for (m,_) in crumb2])
                    path_cells.extend([cell for (_,cell) in crumb2])
                    cur_cell = path_cells[-1]; cur_xy = cell_to_xy(board, cur_cell, gs)

    path_xy = [cell_to_xy(board, c, gs) for c in path_cells]
    expr = moves_to_expression(path_moves, (shooter.x, shooter.y), gs)

    overlay = roi.copy()
    if obs_contours:
        cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)
    h,w = obs_main.shape[:2]
    cv2.rectangle(overlay, (0,0), (w-1,h-1), (0,255,0), 2)

    for a in actors:
        if a is shooter:
            cv2.circle(overlay, (a.px,a.py), 18, (0,215,255), 3)
            cv2.putText(overlay, "S", (a.px+10,a.py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2, cv2.LINE_AA)
        else:
            color = (0,0,255) if a.side=="enemy" else (255,0,0)
            label = "E" if a.side=="enemy" else "A"
            cv2.circle(overlay, (a.px,a.py), 6, color, 2)
            cv2.putText(overlay, label, (a.px+8,a.py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if a.side == "enemy":
            rpx = int((ENEMY_RADIUS / (board.x_range[1]-board.x_range[0])) * board.w)
            cv2.circle(overlay, (a.px,a.py), max(6,rpx), (0,140,255), 1)

    if len(path_xy) >= 2:
        pts_px = [board.xy_to_px(x,y) for (x,y) in path_xy]
        for i in range(1, len(pts_px)):
            cv2.line(overlay, pts_px[i-1], pts_px[i], (0,0,255), 2)

    if overlay_path:
        try: cv2.imwrite(overlay_path, overlay)
        except Exception: pass

    # Build occ from original (tight) obs for display (light-green layer)
    occ_main = make_occ_grid(board, obs_main, gs)

    allies_cells = []
    enemies_cells = []
    shooter_cell = None
    for a in actors:
        cell = xy_to_cell(board, (a.x, a.y), gs, occ)
        if a is shooter:
            shooter_cell = cell
        elif a.side == "enemy":
            enemies_cells.append(cell)
        else:
            allies_cells.append(cell)

    crit = _draw_critical_grid(
        occ_plan=occ,
        occ_main=occ_main,
        path_cells=path_cells,
        actor_cells={
            "allies": allies_cells,
            "enemies": enemies_cells,
            "shooter": shooter_cell
        },
        scale=10,
    )

    return {"expr": expr if expr.strip() else "0", "overlay_bgr": overlay, "crit_overlay_bgr": crit}
