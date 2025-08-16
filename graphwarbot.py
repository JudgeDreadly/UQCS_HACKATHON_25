#!/usr/bin/env python3
"""
Graphwar Solver (auto-capture, full overlay, multi-bridge)
----------------------------------------------------------
- ALWAYS captures the primary monitor (mss)
- Detects the playfield, obstacles (mask + contours), players
- Identifies the current shooter by red aura (filters out name-tag badges)
- Plans a single-valued path (monotone in x) that visits enemies on one side
  while avoiding obstacles (grid A*). Pins the polyline through enemy points.
- Converts the (simplified) path into a sum of bridge primitives:
    Diagonal: a*(|x+b| - |x+c|)
    Step:     k/(1+exp(-a*(x+c)))   [a=55]
- Overlay shows EXACTLY what the solver believes (actors, obstacles, chosen path).
- Prints the final expression to paste in Graphwar.

Dependencies:
    pip install opencv-python mss numpy
"""
import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# -------- Screenshot (always-on) --------
def capture_fullscreen(output_path: str = "graphwar_capture.png") -> str:
    """Capture the primary monitor and save to output_path."""
    try:
        import mss, mss.tools
    except Exception as e:
        raise RuntimeError("Screenshot capture requires the 'mss' package. Install with: pip install mss") from e
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])  # primary monitor
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)
    return output_path
# ----------------------------------------

@dataclass
class Board:
    x0: int
    y0: int
    w: int
    h: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def px_to_xy(self, px: float, py: float) -> Tuple[float, float]:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        x = x_min + (px / self.w) * (x_max - x_min)
        y = y_max - (py / self.h) * (y_max - y_min)  # image y down, graph y up
        return x, y

    def xy_to_px(self, x: float, y: float) -> Tuple[int, int]:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        px = int(round((x - x_min) / (x_max - x_min) * self.w))
        py = int(round((y_max - y) / (y_max - y_min) * self.h))
        return px, py

@dataclass
class Actor:
    x: float
    y: float
    side: str   # 'ally' | 'enemy' | 'unknown'
    px: int
    py: int
    area: int = 0

# ---------- Board ROI ----------
def find_board_roi(img_bgr: np.ndarray) -> Tuple['Board', np.ndarray]:
    if cv2 is None:
        raise RuntimeError("This script requires OpenCV (cv2). Please install opencv-python.")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([179, 45, 255])
    mask_white = cv2.inRange(hsv, lower, upper)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Couldn't find a white board region in the screenshot.")
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img_bgr[y:y+h, x:x+w].copy()
    board = Board(x0=x, y0=y, w=w, h=h, x_range=(-25.0, 25.0), y_range=(-15.0, 15.0))
    return board, roi

# ---------- Name tags ----------
def detect_white_labels(roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Return bounding boxes of white-ish rectangles (player name tags)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 40, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w / max(1, h)
        if 200 <= area <= 8000 and ar >= 1.2:
            boxes.append((x,y,w,h))
    return boxes

def close_to_any_label(px:int, py:int, labels:List[Tuple[int,int,int,int]], max_dist:int=28) -> bool:
    for (x,y,w,h) in labels:
        cx, cy = x + w//2, y + h//2
        if (px - cx)**2 + (py - cy)**2 <= max_dist**2:
            return True
    return False

# ---------- Obstacles (mask + contours) ----------
def _remove_axes(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=int(0.5*mask.shape[1]), maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(mask, (x1,y1), (x2,y2), 0, 5)
    return mask

def obstacle_mask_and_contours(roi_bgr: np.ndarray):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)
    mask = _remove_axes(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= 200]
    mask_clean = np.zeros_like(mask)
    cv2.drawContours(mask_clean, kept, -1, 255, thickness=cv2.FILLED)
    return mask_clean, kept

# ---------- Actors ----------
def detect_actors(roi_bgr: np.ndarray, min_area: int=60) -> List[Actor]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(hsv, np.array([0, 50, 80]), np.array([179, 255, 255]))
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_color, connectivity=8)
    actors: List[Actor] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > 5000:
            continue
        cx, cy = centroids[i]
        actors.append(Actor(x=0.0, y=0.0, side="unknown", px=int(cx), py=int(cy), area=area))
    return actors

def filter_out_name_badges(roi_bgr: np.ndarray, actors: List[Actor]) -> List[Actor]:
    labels = detect_white_labels(roi_bgr)
    keep: List[Actor] = []
    for a in actors:
        if a.area < 200 and close_to_any_label(a.px, a.py, labels, max_dist=32):
            continue
        keep.append(a)
    return keep

def assign_coords(board: Board, actors: List[Actor]) -> None:
    for a in actors:
        a.x, a.y = board.px_to_xy(a.px, a.py)

def choose_soldier_positional(actors: List[Actor], team_hint: str="left") -> Optional[Actor]:
    if not actors:
        return None
    return min(actors, key=lambda t: t.x) if team_hint == "left" else max(actors, key=lambda t: t.x)

def pick_shooter_by_red_aura(roi_bgr: np.ndarray, actors: List[Actor]) -> Optional[Actor]:
    if cv2 is None or not actors:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    low1, high1 = np.array([0,120,120]),   np.array([10,255,255])
    low2, high2 = np.array([170,120,120]), np.array([179,255,255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, low1, high1), cv2.inRange(hsv, low2, high2))
    mask_red = cv2.medianBlur(mask_red, 5)

    labels = detect_white_labels(roi_bgr)
    best = None; best_score = 0
    for a in actors:
        if a.area < 200 and close_to_any_label(a.px, a.py, labels, max_dist=32):
            continue
        rr_inner, rr_outer = 6, 28
        h, w = mask_red.shape[:2]
        y,x = a.py, a.px
        y0 = max(0, y-rr_outer); y1 = min(h, y+rr_outer+1)
        x0 = max(0, x-rr_outer); x1 = min(w, x+rr_outer+1)
        sub = mask_red[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - y)**2 + (xx - x)**2
        donut = (dist2 <= rr_outer**2) & (dist2 >= rr_inner**2)
        score = int((sub > 0)[donut].sum())
        if score > best_score:
            best_score = score; best = a
    return best if (best is not None and best_score >= 20) else None

def classify_sides(actors: List[Actor], soldier: Actor) -> None:
    for a in actors:
        if a is soldier:
            a.side = "ally"
        else:
            a.side = "ally" if (a.x <= 0 and soldier.x <= 0) or (a.x >= 0 and soldier.x >= 0) else "enemy"

# ---------- Bridge math ----------
def diagonal_params_from_line(xs: float, xe: float, slope_m: float) -> Tuple[float, float, float]:
    start, end = (xs, xe) if xs < xe else (xe, xs)
    a = abs(slope_m) / 2.0     # slope = ±2a over [start,end]
    if slope_m >= 0:
        b, c = -start, -end
    else:
        b, c = -end, -start
    return a, b, c

# pick bridge per segment
def segment_to_bridge(x1: float, y1: float, x2: float, y2: float) -> Tuple[str, Dict]:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) >= 0.4:  # regular slanted piece -> diagonal
        m = dy / dx
        a,b,c = diagonal_params_from_line(x1, x2, m)
        term = f"{a:.4f}*(abs(x+({b:+.4f}))-abs(x+({c:+.4f})))"
        return term, {"type":"diagonal","a":a,"b":b,"c":c,"start":x1,"end":x2,"slope":m}
    else:
        # mostly vertical -> step at x1 (positive k raises, negative lowers)
        k = dy
        a = 55.0
        c = -x1
        term = f"{k:.4f}/(1+exp(-{a:.0f}*(x+({c:+.4f}))))"
        return term, {"type":"step","k":k,"a":a,"c":c,"x_at":x1}

def build_expression_from_polyline(pts: List[Tuple[float,float]]) -> Tuple[str, List[Dict]]:
    if len(pts) < 2:
        return "0", []
    parts = []
    meta: List[Dict] = []
    for i in range(1, len(pts)):
        (x1,y1),(x2,y2) = pts[i-1], pts[i]
        if abs(x2-x1) < 1e-9 and abs(y2-y1) < 1e-9:
            continue
        term, info = segment_to_bridge(x1,y1,x2,y2)
        parts.append(term)
        meta.append(info)
    expr = " + ".join(parts)
    # simple cleanups
    expr = expr.replace("+-","-").replace("--","+")
    return expr, meta

# ---------- Grid planning (monotone A*) ----------
def build_occupancy(board: Board, obs_mask: np.ndarray, dx: float=0.25, dy: float=0.25):
    xs = np.arange(board.x_range[0], board.x_range[1] + 1e-6, dx)
    ys = np.arange(board.y_range[0], board.y_range[1] + 1e-6, dy)
    occ = np.zeros((len(ys), len(xs)), dtype=np.uint8)  # 1=blocked
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            px, py = board.xy_to_px(x, y)
            px = int(np.clip(px, 0, board.w-1)); py = int(np.clip(py, 0, board.h-1))
            occ[j, i] = 1 if (obs_mask is not None and obs_mask[py, px] > 0) else 0
    return xs, ys, occ

def nearest_idx(xs, ys, x, y):
    i = int(np.clip(np.searchsorted(xs, x), 0, len(xs)-1))
    j = int(np.clip(np.searchsorted(ys, y), 0, len(ys)-1))
    return i, j

def astar_monotone(xs, ys, occ, start, goal, dir_sign: int):
    """A* that only moves forward in x (dir_sign=+1 right, -1 left)."""
    from heapq import heappush, heappop
    si, sj = start; gi, gj = goal
    W, H = len(xs), len(ys)
    moves = [(dir_sign, 0), (dir_sign, +1), (dir_sign, -1)]
    g = { (si,sj): 0.0 }
    came = {}
    pq = []
    def h(i,j):  # L1 distance
        return abs(gi - i) + abs(gj - j)
    heappush(pq, (h(si,sj), (si,sj)))
    seen=set()
    while pq:
        _, (i,j) = heappop(pq)
        if (i,j) in seen: 
            continue
        seen.add((i,j))
        if i == gi and j == gj:
            # reconstruct
            path=[]
            cur=(i,j)
            while cur in came:
                path.append(cur)
                cur=came[cur]
            path.append((si,sj))
            path.reverse()
            return [(xs[ii], ys[jj]) for (ii,jj) in path]
        for di, dj in moves:
            ni, nj = i+di, j+dj
            if ni<0 or ni>=W or nj<0 or nj>=H: 
                continue
            if occ[nj, ni]: 
                continue
            if (dir_sign==+1 and ni<i) or (dir_sign==-1 and ni>i):
                continue
            newg = g[(i,j)] + 1.0
            if (ni,nj) not in g or newg < g[(ni,nj)]:
                g[(ni,nj)] = newg
                came[(ni,nj)] = (i,j)
                heappush(pq, (newg + h(ni,nj), (ni,nj)))
    return []  # no path

def plan_path(board: Board, obs_mask: np.ndarray, soldier_xy: Tuple[float,float],
              enemies_xy: List[Tuple[float,float]], direction: str) -> List[Tuple[float,float]]:
    """Plan path from shooter to every enemy on one side (x-order, monotone)."""
    if not enemies_xy:
        return []
    dir_sign = +1 if direction=="right" else -1
    xs, ys, occ = build_occupancy(board, obs_mask, dx=0.25, dy=0.25)
    sx, sy = soldier_xy
    path_total: List[Tuple[float,float]] = []
    curx, cury = sx, sy
    # order targets
    enemies_sorted = sorted(enemies_xy, key=lambda p: p[0], reverse=(dir_sign==-1))
    for (tx, ty) in enemies_sorted:
        si, sj = nearest_idx(xs, ys, curx, cury)
        gi, gj = nearest_idx(xs, ys, tx, ty)
        seg = astar_monotone(xs, ys, occ, (si,sj), (gi,gj), dir_sign)
        if not seg:
            continue
        if path_total:
            seg = seg[1:]  # drop duplicate join
        path_total.extend(seg)
        # pin EXACT enemy point
        if not path_total or (abs(path_total[-1][0]-tx)>1e-6 or abs(path_total[-1][1]-ty)>1e-6):
            path_total.append((tx, ty))
        curx, cury = tx, ty
    # ensure unique and strictly increasing x (within direction)
    cleaned = []
    lastx = None
    for (x,y) in path_total:
        if lastx is not None and abs(x-lastx) < 1e-6:
            continue
        cleaned.append((x,y))
        lastx = x
    return cleaned

# ---------- Simplify + Expression ----------
def simplify_polyline(pl: List[Tuple[float,float]], slope_tol: float=0.08) -> List[Tuple[float,float]]:
    if len(pl) <= 2: return pl
    simp = [pl[0]]
    for p in pl[1:]:
        if abs(p[0]-simp[-1][0]) < 1e-9 and abs(p[1]-simp[-1][1]) < 1e-9:
            continue
        if len(simp) >= 2:
            m_prev = (simp[-1][1]-simp[-2][1])/(simp[-1][0]-simp[-2][0] + 1e-9)
            m_new  = (p[1]-simp[-1][1])/(p[0]-simp[-1][0] + 1e-9)
            if abs(m_new - m_prev) < slope_tol:
                simp[-1] = p
                continue
        simp.append(p)
    return simp

# ---------- Main solver ----------
def solve(image_path: str, team_hint: str="left",
          x_range: Tuple[float,float]=(-25.0,25.0), y_range: Tuple[float,float]=(-15.0,15.0),
          tolerance: float=0.75, min_area: int=60, debug_out: Optional[str]=None) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to run this script.")
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read screenshot: {image_path}")
    board, roi = find_board_roi(bgr)
    board.x_range = x_range; board.y_range = y_range

    obs_mask, obs_contours = obstacle_mask_and_contours(roi)

    actors_all = detect_actors(roi, min_area=min_area)
    actors = filter_out_name_badges(roi, actors_all)
    assign_coords(board, actors)

    # shooter: prefer red aura, otherwise positional hint
    soldier = pick_shooter_by_red_aura(roi, actors) or choose_soldier_positional(actors, team_hint=team_hint)
    if soldier is None:
        raise RuntimeError("No actors detected.")
    classify_sides(actors, soldier)

    enemies = [a for a in actors if a.side == "enemy"]
    allies  = [a for a in actors if a.side == "ally" and a is not soldier]

    # plan path for each side, pin through enemies
    def side_plan(side: str):
        if side=="right":
            targets = [(e.x, e.y) for e in enemies if e.x >= soldier.x]
        else:
            targets = [(e.x, e.y) for e in enemies if e.x <= soldier.x]
        poly = plan_path(board, obs_mask, (soldier.x, soldier.y), targets, side) if targets else []
        # count hits against pinned targets
        hits = 0
        for (tx,ty) in targets:
            if any(abs(px-tx)<=1e-6 and abs(py-ty)<=1e-6 for (px,py) in poly):
                hits += 1
        return poly, hits

    poly_r, hits_r = side_plan("right")
    poly_l, hits_l = side_plan("left")
    if hits_r > hits_l:
        chosen_side, poly = "right", poly_r
    else:
        chosen_side, poly = "left", poly_l

    if not poly:
        # simple fallback: straight segment toward farthest enemy (or 5 units forward)
        far = max(enemies, key=lambda e: abs(e.x - soldier.x)) if enemies else None
        target = (far.x, far.y) if far else (soldier.x + 5.0, soldier.y)
        poly = [(soldier.x, soldier.y), target]

    # simplify & convert to expression
    poly_simplified = simplify_polyline(poly, slope_tol=0.08)
    expr, bridge_meta = build_expression_from_polyline(poly_simplified)

    # ---- Overlay: EXACT solver view ----
    overlay = roi.copy()
    # obstacles
    if obs_contours:
        cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)  # green outline

    # actors
    for a in actors:
        color = (0,0,255) if a.side=="enemy" else (255,128,0) if a is soldier else (255,0,0)  # enemies red, shooter amber ring, allies blue-ish
        base = (0,0,255) if a.side=="enemy" else (255,0,0)
        if a is soldier:
            cv2.circle(overlay, (a.px, a.py), 18, (0,215,255), 3)  # gold ring
            cv2.putText(overlay, "S", (a.px+10, a.py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,215,255), 2, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (a.px, a.py), 6, base, 2)
            cv2.putText(overlay, "E" if a.side=="enemy" else "A", (a.px+8, a.py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base, 1, cv2.LINE_AA)

    # full planned path
    if poly and len(poly) >= 2:
        pts_px = [board.xy_to_px(x,y) for (x,y) in poly]
        for i in range(1, len(pts_px)):
            cv2.line(overlay, pts_px[i-1], pts_px[i], (0,0,255), 2)
    # simplified path used for expression
    if len(poly_simplified) >= 2:
        pts_px_s = [board.xy_to_px(x,y) for (x,y) in poly_simplified]
        for i in range(1, len(pts_px_s)):
            cv2.line(overlay, pts_px_s[i-1], pts_px_s[i], (30,30,200), 2)

    if debug_out:
        cv2.imwrite(debug_out, overlay)

    debug_actors = [{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is soldier)} for a in actors]
    return {
        "expr": expr if expr.strip() else "0",
        "side": chosen_side,
        "soldier": {"x": float(soldier.x), "y": float(soldier.y)},
        "hits_right": hits_r, "hits_left": hits_l,
        "actors": debug_actors,
        "polyline_points": poly_simplified,
        "bridges": bridge_meta,
        "debug_overlay": debug_out or ""
    }

def main():
    parser = argparse.ArgumentParser(description="Graphwar solver (auto-capture, full overlay)")
    parser.add_argument("--team", choices=["left","right"], default="left", help="Which side your team spawns on")
    parser.add_argument("--xrange", nargs=2, type=float, default=[-25.0, 25.0], help="x-min x-max (graph units)")
    parser.add_argument("--yrange", nargs=2, type=float, default=[-15.0, 15.0], help="y-min y-max (graph units)")
    parser.add_argument("--tolerance", type=float, default=0.75, help="hit tolerance in graph units")
    parser.add_argument("--min_area", type=int, default=60, help="min blob area to treat as an actor")
    parser.add_argument("--debug_out", default="graphwar_overlay.png", help="Path to save overlay PNG")
    args = parser.parse_args()

    screenshot_path = capture_fullscreen("graphwar_capture.png")
    res = solve(
        image_path=screenshot_path,
        team_hint=args.team,
        x_range=(args.xrange[0], args.xrange[1]),
        y_range=(args.yrange[0], args.yrange[1]),
        tolerance=args.tolerance,
        min_area=args.min_area,
        debug_out=(args.debug_out if args.debug_out else None)
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
