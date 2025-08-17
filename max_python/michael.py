#!/usr/bin/env python3
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

ENEMY_RADIUS = 0.8   # graph units of "leeway" to count as a hit/target band
X_GROUP_TOL  = 0.6   # enemies whose x differ <= this are grouped into a column

# ---------- Capture ----------
def capture_fullscreen_bgr() -> np.ndarray:
    """Return a BGR numpy image of the primary monitor (no file saved)."""
    try:
        import mss
    except Exception as e:
        raise RuntimeError("Screenshot capture requires: pip install mss") from e
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])  # primary
        frame = np.array(sct_img)            # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# ---------- Data ----------
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
        y = y_max - (py / self.h) * (y_max - y_min)
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

# ---------- ROI ----------
def find_board_roi(img_bgr: np.ndarray) -> Tuple['Board', np.ndarray]:
    if cv2 is None:
        raise RuntimeError("This script requires OpenCV (cv2).")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 45, 255]))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Couldn't find a white board region.")
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img_bgr[y:y+h, x:x+w].copy()
    board = Board(x0=x, y0=y, w=w, h=h, x_range=(-25.0, 25.0), y_range=(-15.0, 15.0))
    return board, roi

# ---------- Name tags ----------
def detect_white_labels(roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 40, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w / max(1, h)
        if 200 <= area <= 8000 and ar >= 1.1:
            boxes.append((x,y,w,h))
    return boxes

def close_to_any_label(px:int, py:int, labels:List[Tuple[int,int,int,int]], max_dist:int=28) -> bool:
    for (x,y,w,h) in labels:
        cx, cy = x + w//2, y + h//2
        if (px - cx)**2 + (py - cy)**2 <= max_dist**2:
            return True
    return False

# ---------- Obstacles ----------
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
    # subtract name tags (dilated)
    boxes = detect_white_labels(roi_bgr)
    label_mask = np.zeros_like(mask)
    for (x,y,w,h) in boxes:
        pad = 10
        cv2.rectangle(label_mask, (max(0,x-pad), max(0,y-pad)),
                      (min(mask.shape[1]-1, x+w+pad), min(mask.shape[0]-1, y+h+pad)),
                      255, thickness=cv2.FILLED)
    mask[label_mask > 0] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= 200]
    mask_clean = np.zeros_like(mask)
    cv2.drawContours(mask_clean, kept, -1, 255, thickness=cv2.FILLED)
    return mask_clean, kept

def add_border_as_obstacle(mask: np.ndarray, border_px: int = 8) -> np.ndarray:
    h, w = mask.shape[:2]
    border = np.zeros_like(mask)
    cv2.rectangle(border, (0,0), (w-1,h-1), 255, thickness=border_px)
    return cv2.bitwise_or(mask, border)

def inflate_for_safety(mask: np.ndarray, px: int = 2) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    return cv2.dilate(mask, k, iterations=1)

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

def classify_by_center(actors: List[Actor]) -> None:
    for a in actors:
        a.side = "ally" if a.x < 0 else "enemy"

# ---------- Red aura detection ----------
def red_mask(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    low1, high1 = np.array([0,  60, 70]),  np.array([12,255,255])
    low2, high2 = np.array([168,60, 70]),  np.array([179,255,255])
    m = cv2.bitwise_or(cv2.inRange(hsv, low1, high1), cv2.inRange(hsv, low2, high2))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return m

def choose_shooter_by_red_ring(roi_bgr: np.ndarray, actors: List[Actor]) -> Optional[Actor]:
    if not actors:
        return None
    rm = red_mask(roi_bgr)
    if int((rm>0).sum()) < 20:
        return None

    best = None
    best_score = 0.0
    R_IN, R_OUT = 6, 30
    NBINS = 36

    for a in actors:
        h, w = rm.shape
        y, x = a.py, a.px
        y0, y1 = max(0, y-R_OUT), min(h, y+R_OUT+1)
        x0, x1 = max(0, x-R_OUT), min(w, x+R_OUT+1)
        sub = rm[y0:y1, x0:x1]
        if sub.size == 0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (yy - y)**2 + (xx - x)**2
        donut = (d2 <= R_OUT**2) & (d2 >= R_IN**2)
        reds = (sub > 0) & donut
        cnt = int(reds.sum())
        if cnt == 0:
            continue

        ys, xs = np.nonzero(reds)
        ang = np.arctan2((ys + y0) - y, (xs + x0) - x)
        bins = ((ang + np.pi) / (2*np.pi) * NBINS).astype(int)
        bins = np.clip(bins, 0, NBINS-1)
        coverage = np.unique(bins).size / NBINS
        score = cnt * (coverage**1.5)
        if coverage >= 0.25 and score > best_score:
            best_score = score
            best = a
    return best

def choose_leftmost(actors: List[Actor]) -> Optional[Actor]:
    return min(actors, key=lambda t: t.x) if actors else None

# ---------- Bridge math ----------
def diagonal_params_from_line(xs: float, xe: float, slope_m: float) -> Tuple[float, float, float]:
    start, end = (xs, xe) if xs < xe else (xe, xs)
    a = abs(slope_m) / 2.0
    if slope_m >= 0:
        b, c = -start, -end
    else:
        b, c = -end, -start
    return a, b, c

def segment_to_bridge(x1: float, y1: float, x2: float, y2: float) -> Tuple[str, Dict]:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) >= 0.45:  # slanted piece -> diagonal
        m = dy / dx if dx != 0 else 0.0
        a,b,c = diagonal_params_from_line(x1, x2, m)
        term = f"{a:.4f}*(abs(x+({b:+.4f}))-abs(x+({c:+.4f})))"
        return term, {"type":"diagonal","a":a,"b":b,"c":c,"start":x1,"end":x2,"slope":m}
    else:               # near-vertical -> step
        k = dy
        a = 55.0
        c = -x1
        term = f"{k:.4f}/(1+exp(-{a:.0f}*(x+({c:+.4f}))))"
        return term, {"type":"step","k":k,"a":a,"c":c,"x_at":x1}

def build_expression_from_polyline(pts: List[Tuple[float,float]]) -> Tuple[str, List[Dict]]:
    if len(pts) < 2: return "0", []
    parts, meta = [], []
    for i in range(1, len(pts)):
        (x1,y1),(x2,y2) = pts[i-1], pts[i]
        if abs(x2-x1) < 1e-9 and abs(y2-y1) < 1e-9: continue
        term, info = segment_to_bridge(x1,y1,x2,y2)
        parts.append(term); meta.append(info)
    expr = " + ".join(parts)
    return expr.replace("+-","-").replace("--","+"), meta

# ---------- RDP with anchors ----------
def rdp_segment(points: List[Tuple[float,float]], eps: float) -> List[Tuple[float,float]]:
    if len(points) < 3:
        return points[:]
    def dist(p, a, b):
        (x,y),(x1,y1),(x2,y2) = p,a,b
        if x1==x2 and y1==y2: return math.hypot(x-x1, y-y1)
        t = ((x-x1)*(x2-x1)+(y-y1)*(y2-y1))/((x2-x1)**2+(y2-y1)**2)
        t = max(0.0, min(1.0, t))
        xp, yp = x1 + t*(x2-x1), y1 + t*(y2-y1)
        return math.hypot(x-xp, y-yp)
    a, b = points[0], points[-1]
    idx, dmax = 0, 0.0
    for i in range(1, len(points)-1):
        d = dist(points[i], a, b)
        if d > dmax: idx, dmax = i, d
    if dmax > eps:
        left = rdp_segment(points[:idx+1], eps)
        right = rdp_segment(points[idx:], eps)
        return left[:-1] + right
    else:
        return [a, b]

def rdp_with_anchors(points: List[Tuple[float,float]], anchor_idx: List[int], eps: float) -> List[Tuple[float,float]]:
    """Simplify while forcing anchors to remain."""
    anchor_idx = sorted(set([max(0,min(len(points)-1,i)) for i in anchor_idx]))
    if not anchor_idx: return rdp_segment(points, eps)
    out = []
    for a,b in zip(anchor_idx[:-1], anchor_idx[1:]):
        seg = points[a:b+1]
        simp = rdp_segment(seg, eps)
        if out:
            out.extend(simp[1:])
        else:
            out.extend(simp)
    return out

# ---------- Grid + A* ----------
def build_occupancy_and_cost(board: Board, obs_mask: np.ndarray,
                             dx: float=0.20, dy: float=0.20, w_prox: float=4.0):
    xs = np.arange(board.x_range[0], board.x_range[1] + 1e-9, dx)
    ys = np.arange(board.y_range[0], board.y_range[1] + 1e-9, dy)
    free = (obs_mask == 0).astype(np.uint8)*255
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    dist_norm = dist / dist.max() if dist.max() > 0 else dist
    occ  = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    cost = np.zeros_like(occ, dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            px, py = board.xy_to_px(x, y)
            px = int(np.clip(px, 0, board.w-1)); py = int(np.clip(py, 0, board.h-1))
            blocked = obs_mask[py, px] > 0
            occ[j, i] = 1 if blocked else 0
            prox = 1.0 - float(dist_norm[py, px])  # 0 far, 1 near
            cost[j, i] = 1.0 + w_prox * (prox**2)
    return xs, ys, occ, cost

def nearest_idx(xs, ys, x, y):
    i = int(np.clip(np.searchsorted(xs, x), 0, len(xs)-1))
    j = int(np.clip(np.searchsorted(ys, y), 0, len(ys)-1))
    return i, j

def snap_to_free(occ: np.ndarray, i: int, j: int, max_r: int = 6) -> Tuple[int,int]:
    if 0 <= j < occ.shape[0] and 0 <= i < occ.shape[1] and occ[j, i] == 0:
        return i, j
    best = None; bestd = 1e9
    for r in range(1, max_r+1):
        for dj in range(-r, r+1):
            for di in (-r, r):
                ni, nj = i+di, j+dj
                if 0 <= nj < occ.shape[0] and 0 <= ni < occ.shape[1] and occ[nj, ni] == 0:
                    d = di*di + dj*dj
                    if d < bestd: bestd, best = d, (ni, nj)
        for di in range(-r+1, r):
            for dj in (-r, r):
                ni, nj = i+di, j+dj
                if 0 <= nj < occ.shape[0] and 0 <= ni < occ.shape[1] and occ[nj, ni] == 0:
                    d = di*di + dj*dj
                    if d < bestd: bestd, best = d, (ni, nj)
        if best is not None: return best
    ys, xs = np.where(occ == 0)
    if len(xs):
        k = int(np.argmin((xs - i)**2 + (ys - j)**2))
        return int(xs[k]), int(ys[k])
    return i, j

def astar_monotone(xs, ys, occ, cost, start, goal, dir_sign: int, w_turn: float=0.12):
    from heapq import heappush, heappop
    si, sj = start; gi, gj = goal
    W, H = len(xs), len(ys)
    moves = [(dir_sign, 0), (dir_sign, +1), (dir_sign, -1), (0, +1), (0, -1)]
    g = { (si,sj): 0.0 }
    came = {}
    pq = []
    def h(i,j): return abs(gi - i) + abs(gj - j)
    heappush(pq, (h(si,sj), (si,sj), (0,0)))
    seen=set()
    while pq:
        _, (i,j), prev = heappop(pq)
        if (i,j) in seen: continue
        seen.add((i,j))
        if i == gi and j == gj:
            path=[]; cur=(i,j)
            while cur in came:
                path.append(cur); cur=came[cur]
            path.append((si,sj)); path.reverse()
            return [(xs[ii], ys[jj]) for (ii,jj) in path]
        for di, dj in moves:
            ni, nj = i+di, j+dj
            if ni<0 or ni>=W or nj<0 or nj>=H: continue
            if occ[nj, ni]: continue
            if (dir_sign==+1 and ni<i) or (dir_sign==-1 and ni>i): continue
            step = cost[nj, ni]
            if prev != (0,0) and (di, dj) != prev: step += w_turn
            newg = g[(i,j)] + float(step)
            if (ni,nj) not in g or newg < g[(ni,nj)]:
                g[(ni,nj)] = newg; came[(ni,nj)] = (i,j)
                heappush(pq, (newg + h(ni,nj), (ni,nj), (di,dj)))
    return []

# ---------- Enemy grouping by x ----------
def group_enemies_by_x(enemies_xy: List[Tuple[float,float]], start_x: float, x_tol: float=X_GROUP_TOL):
    """Return list of columns; each column is a list of (x,y). Only enemies with x >= start_x."""
    cand = [p for p in enemies_xy if p[0] >= start_x]
    cand.sort(key=lambda p: p[0])
    cols: List[List[Tuple[float,float]]] = []
    cur: List[Tuple[float,float]] = []
    cur_x = None
    for x,y in cand:
        if cur_x is None or abs(x - cur_x) <= x_tol:
            cur.append((x,y))
            cur_x = x if cur_x is None else (cur_x + x)/2.0
        else:
            cols.append(cur); cur = [(x,y)]; cur_x = x
    if cur: cols.append(cur)
    return cols

# ---------- Plan path (anchors recorded) ----------
def plan_path(board: Board, obs_mask: np.ndarray, soldier_xy: Tuple[float,float],
              enemies_xy: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], List[int]]:
    xs, ys, occ, cost = build_occupancy_and_cost(board, obs_mask, dx=0.20, dy=0.20, w_prox=4.0)
    sx, sy = soldier_xy
    cols = group_enemies_by_x(enemies_xy, start_x=sx, x_tol=X_GROUP_TOL)

    path: List[Tuple[float,float]] = [(sx, sy)]
    anchors: List[int] = [0]  # keep start
    curx, cury = sx, sy

    for col in cols:
        # one x for the column (max x so we never go backwards)
        xg = max(p[0] for p in col)
        ig, _ = nearest_idx(xs, ys, xg, cury)
        # visit enemies in this column by nearest vertical order
        remaining = col[:]
        while remaining:
            remaining.sort(key=lambda p: abs(p[1] - cury))
            tx, ty = remaining.pop(0)
            si, sj = nearest_idx(xs, ys, curx, cury)
            gi, gj = ig, nearest_idx(xs, ys, xg, ty)[1]
            si, sj = snap_to_free(occ, si, sj, max_r=6)
            gi, gj = snap_to_free(occ, gi, gj, max_r=6)
            seg = astar_monotone(xs, ys, occ, cost, (si,sj), (gi,gj), dir_sign=+1, w_turn=0.12)
            if seg:
                if path: seg = seg[1:]
                path.extend(seg)
                anchors.append(len(path)-1)  # anchor at this target
                curx, cury = path[-1]
            else:
                # as a fallback, just “teleport” anchor to desired grid node
                path.append((xs[gi], ys[gj])); anchors.append(len(path)-1)
                curx, cury = xs[gi], ys[gj]

    return path, anchors

# ---------- Solve ----------
def solve_from_bgr(bgr: np.ndarray,
                   x_range: Tuple[float,float]=(-25.0,25.0), y_range: Tuple[float,float]=(-15.0,15.0),
                   min_area: int=60, overlay_path: Optional[str]=None) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required.")
    board, roi = find_board_roi(bgr)
    board.x_range = x_range; board.y_range = y_range

    obs_mask_raw, obs_contours = obstacle_mask_and_contours(roi)
    obs_mask_with_border = add_border_as_obstacle(obs_mask_raw, border_px=8)
    obs_mask_plan = inflate_for_safety(obs_mask_with_border, px=2)

    actors_all = detect_actors(roi, min_area=min_area)
    actors = filter_out_name_badges(roi, actors_all)
    assign_coords(board, actors)
    classify_by_center(actors)

    shooter = choose_shooter_by_red_ring(roi, actors)
    if shooter is None:
        left_allies = [a for a in actors if a.side == "ally"]
        shooter = choose_leftmost(left_allies) or choose_leftmost(actors)
    if shooter is None:
        raise RuntimeError("No actors detected.")

    enemies_xy = [(a.x, a.y) for a in actors if a.side == "enemy"]

    raw_path, anchor_idx = plan_path(board, obs_mask_plan, (shooter.x, shooter.y), enemies_xy)
    # simplify but keep anchors (enemies)
    path = rdp_with_anchors(raw_path, anchor_idx, eps=0.40)
    expr, _ = build_expression_from_polyline(path)

    # Overlay (only one path)
    if overlay_path:
        overlay = roi.copy()
        if obs_contours:
            cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)  # obstacles (green)
        h, w = obs_mask_with_border.shape[:2]
        cv2.rectangle(overlay, (0,0), (w-1,h-1), (0,255,0), 2)        # border obstacle

        # draw actors + enemy hit radii
        for a in actors:
            if a is shooter:
                cv2.circle(overlay, (a.px, a.py), 18, (0,215,255), 3)
                cv2.putText(overlay, "S", (a.px+10, a.py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2, cv2.LINE_AA)
            else:
                color = (0,0,255) if a.side=="enemy" else (255,0,0)
                label = "E" if a.side=="enemy" else "A"
                cv2.circle(overlay, (a.px, a.py), 6, color, 2)
                cv2.putText(overlay, label, (a.px+8, a.py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # hit radius for enemies
            if a.side == "enemy":
                rpx = int((ENEMY_RADIUS / (board.x_range[1]-board.x_range[0])) * board.w)
                cv2.circle(overlay, (a.px, a.py), max(6,rpx), (0,140,255), 1)

        # final path only
        if len(path) >= 2:
            pts_px = [board.xy_to_px(x,y) for (x,y) in path]
            for i in range(1, len(pts_px)):
                cv2.line(overlay, pts_px[i-1], pts_px[i], (0,0,255), 2)

        cv2.imwrite(overlay_path, overlay)

    out_actors = [{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
    return {"actors": out_actors, "expr": expr if expr.strip() else "0"}

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Graphwar solver (single-path overlay)")
    parser.add_argument("--xrange", nargs=2, type=float, default=[-25.0, 25.0], help="x-min x-max (graph units)")
    parser.add_argument("--yrange", nargs=2, type=float, default=[-15.0, 15.0], help="y-min y-max (graph units)")
    parser.add_argument("--min_area", type=int, default=60, help="min blob area to treat as an actor")
    parser.add_argument("--debug_out", default="graphwar_overlay.png", help="Path to save overlay PNG")
    args = parser.parse_args()

    bgr = capture_fullscreen_bgr()
    res = solve_from_bgr(
        bgr,
        x_range=(args.xrange[0], args.xrange[1]),
        y_range=(args.yrange[0], args.yrange[1]),
        min_area=args.min_area,
        overlay_path=(args.debug_out if args.debug_out else None)
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()