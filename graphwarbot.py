#!/usr/bin/env python3
# STEP-ONLY Graphwar solver (enemy-first; no obstacle crossing; multiple hearts)
# -----------------------------------------------------------------------------
# - Auto-captures the primary monitor.
# - Detects board, obstacles (labels filtered), allies/enemies, and shooter.
# - Plans a staircase path that visits enemies in ascending X:
#     go to (x_pre, y_enemy) then cross to (x_post, y_enemy)
#   Path uses A* with moves {Right, Up, Down} only → realizable via STEP terms.
# - Converts the staircase to exact STEP primitives:
#     k/(1+exp(-a*(x - x_step)))  with a=69 and k = exact Δy
#   Each step is placed inside the following horizontal run at the clearest x,
#   and a collision “tube” around the sigmoid is checked against obstacles.
# - Overlay draws y = f(x) - f(xs) + ys (Graphwar’s auto-translation), so it
#   matches the in-game curve exactly.
# - If --flex, places multiple asymmetric hearts on long flat spans:
#     width: ±3 in x, height: +2 up, -4 down; skips if an ally is inside.
#   Overlay writes “Heart” where each will appear.
#
# Output: JSON -> {"actors":[...], "expr":"..."} and one overlay image.

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# ---------------- tunables ----------------
ENEMY_RADIUS    = 0.8
RDP_EPS         = 0.45

GRID_DX         = 0.20
GRID_DY         = 0.20

TURN_PENALTY    = 0.20
ORIENT_PENALTY_VERT = 0.20
ORIENT_BONUS_HORZ   = 0.15

BORDER_THICK_PX = 8
INFLATE_PX      = 2
INFLATE_PX_LO   = 1
SHRINK_STEPS    = 6
MIN_THICK_PX    = 6.0

# logistic STEP
STEP_STEEPNESS       = 69.0
STEP_EPS_DY          = 0.30
STEP_OFFSET_FIRST    = 0.25  # first step offset so shooter isn’t exactly on it
STEP_OFFSET          = 0.12
STEP_MIN_GAP_AFTER   = 0.05
STEP_MAX_OFFSET_FRAC = 0.33
MERGE_DX             = 0.60

# “near” obstacle band for cost; prevents over-penalizing open space
PROX_TAU_PX     = 18.0
PROX_WEIGHT     = 6.0

BEST_EFFORT_MAX_EXPANSIONS = 120000

# enemy crossing window
X_PRE_EPS       = 0.35
X_POST_EPS      = 0.25

# heart packing (asymmetric footprint around center a at plateau y)
HEART_HALF_W   = 3.0
HEART_UP       = 2.0
HEART_DOWN     = 4.0
HEART_GAP      = 0.6  # min gap from edges/other hearts
# ------------------------------------------

# ---------- capture ----------
def capture_fullscreen_bgr() -> np.ndarray:
    try:
        import mss
    except Exception as e:
        raise RuntimeError("Screenshot capture requires: pip install mss") from e
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        frame = np.array(sct_img)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# ---------- data ----------
@dataclass
class Board:
    x0: int; y0: int; w: int; h: int
    x_range: Tuple[float, float]; y_range: Tuple[float, float]
    def px_to_xy(self, px: float, py: float) -> Tuple[float, float]:
        x_min, x_max = self.x_range; y_min, y_max = self.y_range
        x = x_min + (px / self.w) * (x_max - x_min)
        y = y_max - (py / self.h) * (y_max - y_min)
        return x, y
    def xy_to_px(self, x: float, y: float) -> Tuple[int, int]:
        x_min, x_max = self.x_range; y_min, y_max = self.y_range
        px = int(round((x - x_min) / (x_max - x_min) * self.w))
        py = int(round((y_max - y) / (y_max - y_min) * self.h))
        return px, py

@dataclass
class Actor:
    x: float; y: float; side: str; px: int; py: int; area: int = 0

# ---------- board ROI ----------
def find_board_roi(img_bgr: np.ndarray) -> Tuple['Board', np.ndarray]:
    if cv2 is None:
        raise RuntimeError("Requires OpenCV.")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 45, 255]))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Couldn't find the board.")
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img_bgr[y:y+h, x:x+w].copy()
    return Board(x0=x, y0=y, w=w, h=h, x_range=(-25.0,25.0), y_range=(-15.0,15.0)), roi

# ---------- label cleanup ----------
def detect_white_labels(roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 120, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w/max(1,h)
        if 100<=area<=20000 and ar>=1.03:
            boxes.append((x,y,w,h))
    return boxes

def remove_thin_dark_near_labels(obstacles_mask: np.ndarray, roi_bgr: np.ndarray,
                                 pad:int=24, dilate:int=50) -> np.ndarray:
    boxes = detect_white_labels(roi_bgr)
    if not boxes:
        return obstacles_mask
    er = cv2.erode(obstacles_mask, np.ones((3,3), np.uint8), iterations=1)
    thin = cv2.bitwise_and(obstacles_mask, cv2.bitwise_not(er))
    region = np.zeros_like(obstacles_mask)
    for (x,y,w,h) in boxes:
        x0=max(0,x-pad); y0=max(0,y-pad)
        x1=min(obstacles_mask.shape[1]-1, x+w+pad); y1=min(obstacles_mask.shape[0]-1, y+h+pad)
        cv2.rectangle(region,(x0,y0),(x1,y1),255,thickness=cv2.FILLED)
    if dilate>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate+1,2*dilate+1))
        region=cv2.dilate(region,k,iterations=1)
    cleaned = obstacles_mask.copy()
    cleaned[(region>0) & (thin>0)] = 0
    return cleaned

# ---------- obstacles ----------
def _remove_axes(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=int(0.5*mask.shape[1]), maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(mask, (x1,y1), (x2,y2), 0, 5)
    return mask

def _keep_by_thickness(shape_mask: np.ndarray, min_thick_px: float=MIN_THICK_PX) -> np.ndarray:
    num, labels = cv2.connectedComponents(shape_mask)
    keep = np.zeros_like(shape_mask)
    for i in range(1, num):
        comp = np.uint8(labels == i) * 255
        area = int((comp > 0).sum())
        if area < 80:  # specks
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * cv2.contourArea(cnt) / (per * per + 1e-9)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max())
        if thick >= min_thick_px or (circularity >= 0.60 and area >= 150):
            keep = cv2.bitwise_or(keep, comp)
    return keep

def obstacle_mask_and_contours(roi_bgr: np.ndarray):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mask = np.uint8((v < 60) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    mask = _remove_axes(mask)
    mask = remove_thin_dark_near_labels(mask, roi_bgr, pad=24, dilate=50)
    mask = _keep_by_thickness(mask, MIN_THICK_PX)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, contours, -1, 255, thickness=cv2.FILLED)
    return clean, contours

def add_border_mask(shape: Tuple[int,int], px:int=BORDER_THICK_PX)->np.ndarray:
    h,w = shape
    border = np.zeros((h,w), dtype=np.uint8)
    cv2.rectangle(border,(0,0),(w-1,h-1),255,thickness=px)
    return border

def inflate(mask: np.ndarray, px:int)->np.ndarray:
    if px<=0: return mask.copy()
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px))
    return cv2.dilate(mask,k,iterations=1)

def shrink_obstacles_only(obs_no_border: np.ndarray, steps:int)->List[np.ndarray]:
    out=[]; cur=obs_no_border.copy()
    if steps<=0: return out
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for _ in range(steps):
        cur = cv2.erode(cur, k, iterations=1)
        out.append(cur.copy())
    return out

# ---------- actors ----------
def detect_actors(roi_bgr: np.ndarray, min_area:int=60)->List[Actor]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,50,80]), np.array([179,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    num,_,stats,cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out=[]
    for i in range(1,num):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<min_area or area>5000: continue
        cx,cy=cent[i]
        out.append(Actor(x=0.0,y=0.0,side="unknown",px=int(cx),py=int(cy),area=area))
    return out

def assign_coords(board: Board, actors: List[Actor])->None:
    for a in actors:
        a.x,a.y = board.px_to_xy(a.px,a.py)

def classify_by_center(actors: List[Actor])->None:
    for a in actors:
        a.side = "ally" if a.x<0 else "enemy"

# ---------- shooter ----------
def red_mask(roi_bgr: np.ndarray)->np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,60,70]),   np.array([12,255,255]))
    m2 = cv2.inRange(hsv, np.array([168,60,70]), np.array([179,255,255]))
    m  = cv2.bitwise_or(m1,m2)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), iterations=1)
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    return m

def choose_shooter_by_red_ring(roi_bgr: np.ndarray, actors: List[Actor])->Optional[Actor]:
    if not actors: return None
    rm = red_mask(roi_bgr)
    if int((rm>0).sum())<20: return None
    best=None; best_score=0.0; R_IN,R_OUT=6,30; NBINS=36
    for a in actors:
        h,w = rm.shape; y,x=a.py,a.px
        y0,y1=max(0,y-R_OUT),min(h,y+R_OUT+1); x0,x1=max(0,x-R_OUT),min(w,x+R_OUT+1)
        sub = rm[y0:y1,x0:x1]
        if sub.size==0: continue
        yy,xx = np.ogrid[y0:y1, x0:x1]
        d2 = (yy-y)**2 + (xx-x)**2
        reds = (sub>0) & (d2<=R_OUT**2) & (d2>=R_IN**2)
        cnt = int(reds.sum())
        if cnt==0: continue
        ys,xs = np.nonzero(reds)
        ang = np.arctan2((ys+y0)-y, (xs+x0)-x)
        bins = ((ang+np.pi)/(2*np.pi)*NBINS).astype(int)
        bins = np.clip(bins,0,NBINS-1)
        cov = np.unique(bins).size/NBINS
        score = cnt*(cov**1.5)
        if cov>=0.25 and score>best_score:
            best_score=score; best=a
    return best

def choose_leftmost(actors: List[Actor])->Optional[Actor]:
    return min(actors, key=lambda t:t.x) if actors else None

# ---------- grid / cost ----------
def build_occupancy_and_cost(board: Board, obs_mask: np.ndarray,
                             dx: float=GRID_DX, dy: float=GRID_DY,
                             w_prox: float=PROX_WEIGHT, tau_px: float=PROX_TAU_PX):
    xs = np.arange(board.x_range[0], board.x_range[1]+1e-9, dx)
    ys = np.arange(board.y_range[0], board.y_range[1]+1e-9, dy)

    free = (obs_mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)  # px to nearest obstacle/border

    # penalize only when near obstacles
    near = np.clip((tau_px - dist) / max(1.0, tau_px), 0.0, 1.0)
    penal = (near ** 2) * w_prox

    px = np.clip(((xs - board.x_range[0]) / (board.x_range[1]-board.x_range[0]) * (board.w-1)).round().astype(int), 0, board.w-1)
    py = np.clip(((board.y_range[1]-ys) / (board.y_range[1]-board.y_range[0]) * (board.h-1)).round().astype(int), 0, board.h-1)
    occ = (obs_mask[np.ix_(py, px)] > 0).astype(np.uint8)
    cost = 1.0 + penal[np.ix_(py, px)].astype(np.float32)
    return xs,ys,occ,cost,dist

def nearest_idx(xs,ys,x,y):
    i=int(np.clip(np.searchsorted(xs,x),0,len(xs)-1))
    j=int(np.clip(np.searchsorted(ys,y),0,len(ys)-1))
    return i,j

def snap_to_free(occ,i,j,max_r:int=6)->Tuple[int,int]:
    if 0<=j<occ.shape[0] and 0<=i<occ.shape[1] and occ[j,i]==0: return i,j
    best=None; bestd=1e9
    for r in range(1,max_r+1):
        for dj in range(-r,r+1):
            for di in (-r,r):
                ni,nj=i+di,j+dj
                if 0<=nj<occ.shape[0] and 0<=ni<occ.shape[1] and occ[nj,ni]==0:
                    d=di*di+dj*dj
                    if d<bestd: bestd, best=d,(ni,nj)
        for di in range(-r+1,r):
            for dj in (-r,r):
                ni,nj=i+di,j+dj
                if 0<=nj<occ.shape[0] and 0<=ni<occ.shape[1] and occ[nj,ni]==0:
                    d=di*di+dj*dj
                    if d<bestd: bestd, best=d,(ni,nj)
        if best is not None: return best
    ys,xs=np.where(occ==0)
    if len(xs):
        k=int(np.argmin((xs-i)**2+(ys-j)**2))
        return int(xs[k]),int(ys[k])
    return i,j

def astar_monotone(xs,ys,occ,cost,start,goal,dir_sign:int):
    from heapq import heappush, heappop
    si,sj=start; gi,gj=goal
    W,H=len(xs),len(ys)
    moves=[(dir_sign,0),(0,+1),(0,-1)]  # Right/Up/Down only
    g={(si,sj):0.0}; came={}; pq=[]
    def h(i,j): return abs(gi-i)+abs(gj-j)
    heappush(pq,(h(si,sj),(si,sj),(0,0))); seen=set()
    while pq:
        _,(i,j),prev=heappop(pq)
        if (i,j) in seen: continue
        seen.add((i,j))
        if i==gi and j==gj:
            path=[]; cur=(i,j)
            while cur in came:
                path.append(cur); cur=came[cur]
            path.append((si,sj)); path.reverse()
            return [(xs[ii],ys[jj]) for (ii,jj) in path]
        for di,dj in moves:
            ni,nj=i+di,j+dj
            if ni<0 or ni>=W or nj<0 or nj>=H: continue
            if occ[nj,ni]: continue
            if dir_sign==+1 and ni<i: continue
            step = cost[nj,ni]
            if dj != 0: step += ORIENT_PENALTY_VERT
            if di == dir_sign and dj == 0: step = max(0.01, step - ORIENT_BONUS_HORZ)
            if prev!=(0,0) and (di,dj)!=prev: step += TURN_PENALTY
            newg = g[(i,j)] + float(step)
            if (ni,nj) not in g or newg < g[(ni,nj)]:
                g[(ni,nj)] = newg
                came[(ni,nj)] = (i,j)
                heappush(pq,(newg + h(ni,nj),(ni,nj),(di,dj)))
    return []

def best_effort_toward(xs,ys,occ,cost,start,goal,dir_sign:int, max_exp:int=BEST_EFFORT_MAX_EXPANSIONS):
    from heapq import heappush, heappop
    si,sj=start; gi,gj=goal
    W,H=len(xs),len(ys)
    moves=[(dir_sign,0),(0,+1),(0,-1)]
    g={(si,sj):0.0}; came={}; pq=[]
    def h(i,j): return abs(gi-i)+abs(gj-j)
    heappush(pq,(h(si,sj),(si,sj),(0,0))); seen=set()
    best_node=(si,sj); best_f=h(si,sj)
    exp=0
    while pq and exp<max_exp:
        f,(i,j),prev = heappop(pq); exp+=1
        if (i,j) in seen: continue
        seen.add((i,j))
        if f<best_f: best_f=f; best_node=(i,j)
        for di,dj in moves:
            ni,nj=i+di,j+dj
            if ni<0 or ni>=W or nj<0 or nj>=H: continue
            if occ[nj,ni]: continue
            if dir_sign==+1 and ni<i: continue
            step = cost[nj,ni]
            if dj != 0: step += ORIENT_PENALTY_VERT
            if di == dir_sign and dj == 0: step = max(0.01, step - ORIENT_BONUS_HORZ)
            if prev!=(0,0) and (di,dj)!=prev: step += TURN_PENALTY
            newg = g[(i,j)] + float(step)
            if (ni,nj) not in g or newg < g[(ni,nj)]:
                g[(ni,nj)] = newg
                came[(ni,nj)] = (i,j)
                heappush(pq,(newg + h(ni,nj),(ni,nj),(di,dj)))
    path=[]; cur=best_node
    while cur in came:
        path.append(cur); cur=came[cur]
    path.append((si,sj)); path.reverse()
    return [(xs[ii],ys[jj]) for (ii,jj) in path]

# ---------- path planning: enemy-first ----------
def plan_enemy_first(board: Board,
                     obs_mask_main: np.ndarray,
                     soldier_xy: Tuple[float,float],
                     enemies_xy: List[Tuple[float,float]],
                     obs_mask_loose: Optional[np.ndarray]=None,
                     shrink_list: Optional[List[np.ndarray]]=None
                     ) -> Tuple[List[Tuple[float,float]], List[float], np.ndarray]:
    """
    Visit enemies left->right. For each enemy E(xe, ye):
      1) route to (xe - eps_pre, ye)
      2) then cross to (xe + eps_post, ye)
    Returns: grid path, list of all x_pre values (for step bias), dist map.
    """
    _,_,_,_,dist_safe = build_occupancy_and_cost(board, obs_mask_main, dx=GRID_DX, dy=GRID_DY)

    sx,sy=soldier_xy
    targets = sorted(enemies_xy, key=lambda p: p[0])  # left -> right

    path=[(sx,sy)]
    pre_x_list=[]
    curx,cury=sx,sy
    for (xe, ye) in targets:
        x_pre  = xe - X_PRE_EPS
        x_post = xe + X_POST_EPS
        pre_x_list.append(x_pre)

        masks=[obs_mask_main]
        if obs_mask_loose is not None: masks.append(obs_mask_loose)
        if shrink_list: masks.extend(shrink_list)

        # to (x_pre, ye)
        seg1 = try_route_variants(board, masks, (curx,cury), (x_pre, ye), require_progress=True)
        if seg1:
            if path: seg1 = seg1[1:]
            path.extend(seg1); curx,cury = path[-1]
        else:
            # couldn't reach this enemy: skip
            continue

        # cross to (x_post, same y)
        seg2 = try_route_variants(board, masks, (curx,cury), (x_post, cury), require_progress=False)
        if seg2:
            if path: seg2 = seg2[1:]
            path.extend(seg2); curx,cury = path[-1]

    return path, pre_x_list, dist_safe

# ---------- helpers for placing safe steps ----------
STEP_TUBE_MARGIN_Y   = 0.25
STEP_TUBE_XHALF      = 2.0*math.log(50.0)/STEP_STEEPNESS  # ~0.13 when a=69

def _column_clearance_units(board: Board, dist_map: np.ndarray,
                            x_center: float, y1: float, y2: float,
                            half_x: float = 0.12, nx: int = 7, ny: int = 40) -> float:
    xs = np.linspace(x_center - half_x, x_center + half_x, nx)
    ys = np.linspace(min(y1, y2), max(y1, y2), ny)
    min_units = float("inf")
    units_per_px_y = (board.y_range[1] - board.y_range[0]) / board.h
    for xx in xs:
        for yy in ys:
            px, py = board.xy_to_px(xx, yy)
            px = int(np.clip(px, 0, board.w - 1))
            py = int(np.clip(py, 0, board.h - 1))
            d_units = float(dist_map[py, px]) * units_per_px_y
            if d_units < min_units:
                min_units = d_units
    return min_units

def _step_tube_clear(board: Board, obs_mask: np.ndarray,
                     x_step: float, y_before: float, y_after: float,
                     x_half: float = STEP_TUBE_XHALF,
                     margin_y: float = STEP_TUBE_MARGIN_Y) -> bool:
    xa, xb = x_step - x_half, x_step + x_half
    ya, yb = min(y_before, y_after) - margin_y, max(y_before, y_after) + margin_y
    xs = np.linspace(xa, xb, 13)
    ys = np.linspace(ya, yb, 25)
    for yy in ys:
        for xx in xs:
            px, py = board.xy_to_px(xx, yy)
            px = int(np.clip(px, 0, board.w-1))
            py = int(np.clip(py, 0, board.h-1))
            if obs_mask[py, px] > 0:
                return False
    return True

def _choose_clear_step_x(board: Board, dist_map: np.ndarray, obs_mask: np.ndarray,
                         x1: float, y1: float, x2: float, y2: float,
                         base_off: float, prefer_x: Optional[float]) -> float:
    left  = x1 + base_off
    right = max(left + 1e-3, x2 - STEP_MIN_GAP_AFTER)
    if right <= left:
        return left

    cands = np.linspace(left, right, 13).tolist()
    if prefer_x is not None:
        cands += [np.clip(prefer_x, left, right)]
    cands = sorted(set([float(c) for c in cands]))

    best_x = left
    best_score = -1e9
    for xc in cands:
        if not _step_tube_clear(board, obs_mask, xc, y1, y2):
            continue
        clear = _column_clearance_units(board, dist_map, xc, y1, y2, half_x=0.12)
        bias  = -0.03*abs((prefer_x or xc) - xc)
        score = clear + bias
        if score > best_score:
            best_score = score
            best_x = xc

    return best_x

# ---------- grid staircase -> STEP list ----------
def steps_from_grid_path(grid_path: List[Tuple[float,float]],
                         enemy_pre_x: List[float],
                         board: Board,
                         dist_map: np.ndarray,
                         obs_mask: np.ndarray) -> List[Tuple[float,float]]:
    if len(grid_path) < 2: return []
    steps: List[Tuple[float,float]] = []
    i = 0
    while i < len(grid_path)-1:
        x_col = grid_path[i][0]
        y_cur = grid_path[i][1]

        # sum vertical moves at this x
        dy_total = 0.0
        j = i
        while j < len(grid_path)-1 and abs(grid_path[j+1][0] - x_col) < 1e-9:
            dy_total += (grid_path[j+1][1] - grid_path[j][1])
            j += 1

        # next right-run length
        run_dx = 0.0
        while j < len(grid_path)-1 and grid_path[j+1][0] > grid_path[j][0]:
            run_dx += (grid_path[j+1][0] - grid_path[j][0])
            j += 1

        if abs(dy_total) >= STEP_EPS_DY and run_dx > 0:
            base_off = STEP_OFFSET_FIRST if len(steps)==0 else STEP_OFFSET
            base_off = max(base_off, min(STEP_MAX_OFFSET_FRAC*run_dx, run_dx - STEP_MIN_GAP_AFTER))
            prefer = None
            for xp in enemy_pre_x:
                if abs((x_col+run_dx) - xp) < 0.6:
                    prefer = np.clip(xp - 0.18, x_col + STEP_OFFSET, x_col + run_dx - STEP_MIN_GAP_AFTER)
                    break

            x_step = _choose_clear_step_x(board, dist_map, obs_mask,
                                          x_col, y_cur, x_col + run_dx, y_cur + dy_total,
                                          base_off, prefer)
            tries = 0
            while not _step_tube_clear(board, obs_mask, x_step, y_cur, y_cur + dy_total) and tries < 4:
                x_step = min(x_step + 0.08, x_col + run_dx - STEP_MIN_GAP_AFTER)
                tries += 1
            steps.append((x_step, dy_total))

        i = j

    return steps

def merge_steps(steps: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not steps: return steps
    steps = sorted(steps, key=lambda t:t[0])
    merged=[]
    cur_x,cur_k = steps[0]
    for x,k in steps[1:]:
        if (x - cur_x) <= MERGE_DX and (k*cur_k) > 0:
            cur_k += k
            cur_x = max(cur_x, x)
        else:
            if abs(cur_k) >= STEP_EPS_DY:
                merged.append((cur_x, cur_k))
            cur_x,cur_k = x,k
    if abs(cur_k) >= STEP_EPS_DY:
        merged.append((cur_x, cur_k))
    out=[]; last_x=-1e9
    for x,k in merged:
        x = max(x, last_x + 1e-4)
        out.append((x,k)); last_x = x
    return out

# ---------- step math ----------
def build_step_expression(steps: List[Tuple[float,float]]) -> str:
    if not steps:
        return "0"
    parts=[]
    for (x_step,k) in steps:
        parts.append(f"{k:.4f}/(1+exp(-{STEP_STEEPNESS:.0f}*(x-({x_step:.4f}))))")
    expr = " + ".join(parts)
    return expr.replace("+-","-").replace("--","+")

def f_steps(xs: np.ndarray, steps: List[Tuple[float,float]]) -> np.ndarray:
    y = np.zeros_like(xs, dtype=float)
    for (x_step,k) in steps:
        y += k / (1.0 + np.exp(-STEP_STEEPNESS*(xs - x_step)))
    return y

def eval_steps_autotranslated(xs: np.ndarray, steps: List[Tuple[float,float]],
                              xs0: float, y0: float) -> np.ndarray:
    f = f_steps(xs, steps)
    f0 = float(f_steps(np.array([xs0]), steps)[0])
    return f - f0 + y0

def plateaus_from_steps(steps: List[Tuple[float,float]], x0: float, x1: float, y0: float):
    segs=[]
    cur_x=x0; cur_y=y0
    for (xs,k) in steps:
        if xs>cur_x:
            segs.append((cur_x, xs, cur_y))
        cur_y += k
        cur_x = xs
    if cur_x<x1:
        segs.append((cur_x, x1, cur_y))
    return segs

# ---------- hearts ----------
def heart_expr_at(a: float) -> str:
    # exact form supplied by you; center at x=a
    return (
        f"0.4*((abs(x-({a:.4f}))-1.5)-abs(abs(x-({a:.4f}))-1.5))"
        f"+sqrt(2.25-(1.5+0.4*((abs(x-({a:.4f}))-1.5)-abs(abs(x-({a:.4f}))-1.5)))^2)*cos(30*x)"
    )

def _heart_box_clear(board: Board, obs_mask: np.ndarray,
                     a: float, y: float,
                     half_w: float = HEART_HALF_W, up: float = HEART_UP, down: float = HEART_DOWN,
                     margin: float = 0.2) -> bool:
    xs = np.linspace(a - half_w - margin, a + half_w + margin, 18)
    ys = np.linspace(y - down - margin, y + up + margin, 18)
    for yy in ys:
        for xx in xs:
            px,py=board.xy_to_px(xx,yy)
            px = int(np.clip(px,0,board.w-1)); py = int(np.clip(py,0,board.h-1))
            if obs_mask[py,px] > 0:
                return False
    return True

def _ally_in_box(a: float, y: float, allies: List[Tuple[float,float]],
                 half_w: float = HEART_HALF_W, up: float = HEART_UP, down: float = HEART_DOWN,
                 margin: float = 0.2) -> bool:
    L=a-half_w-margin; R=a+half_w+margin
    B=y-down-margin;   T=y+up+margin
    for (ax,ay) in allies:
        if L<=ax<=R and B<=ay<=T:
            return True
    return False

def insert_hearts(board: Board, obs_mask: np.ndarray,
                  steps: List[Tuple[float,float]],
                  x0: float, x1: float, y0: float,
                  overlay: Optional[np.ndarray],
                  ally_pts: List[Tuple[float,float]]) -> str:
    exprs=[]
    segs=plateaus_from_steps(steps, x0, x1, y0)
    stride = 2*HEART_HALF_W + HEART_GAP
    for (L,R,y) in segs:
        usable_L = L + HEART_GAP + HEART_HALF_W
        usable_R = R - HEART_GAP - HEART_HALF_W
        a = usable_L
        while a <= usable_R + 1e-9:
            if (not _ally_in_box(a,y,ally_pts)) and _heart_box_clear(board, obs_mask, a, y):
                exprs.append(heart_expr_at(a))
                if overlay is not None:
                    px,py = board.xy_to_px(a,y)
                    cv2.putText(overlay,"Heart",(px-18,py-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,0,200),2,cv2.LINE_AA)
                a += stride
            else:
                a += 0.5
    return (" + ".join(exprs)).replace("+-","-").replace("--","+")

# ---------- high-level solve ----------
def solve_from_bgr(bgr: np.ndarray,
                   x_range: Tuple[float,float]=(-25.0,25.0),
                   y_range: Tuple[float,float]=(-15.0,15.0),
                   min_area: int=60,
                   overlay_path: Optional[str]=None,
                   flex: bool=False)->dict:
    if cv2 is None: raise RuntimeError("Requires OpenCV.")
    board,roi = find_board_roi(bgr); board.x_range=x_range; board.y_range=y_range

    # obstacles
    obs_raw,obs_contours = obstacle_mask_and_contours(roi)
    border_only = add_border_mask(obs_raw.shape, px=BORDER_THICK_PX)
    obs_main  = cv2.bitwise_or(inflate(obs_raw, INFLATE_PX),  border_only)
    obs_loose = cv2.bitwise_or(inflate(obs_raw, INFLATE_PX_LO), border_only)
    shrinks = shrink_obstacles_only(obs_raw, SHRINK_STEPS)
    shrink_variants = [cv2.bitwise_or(s, border_only) for s in shrinks]

    # actors
    actors = detect_actors(roi, min_area=min_area)
    assign_coords(board, actors); classify_by_center(actors)
    shooter = choose_shooter_by_red_ring(roi, actors)
    if shooter is None:
        left_allies=[a for a in actors if a.side=="ally"]
        shooter = choose_leftmost(left_allies) or choose_leftmost(actors)
    if shooter is None: raise RuntimeError("No actors detected.")
    enemies_xy=[(a.x,a.y) for a in actors if a.side=="enemy"]

    # enemy-first staircase path
    grid_path, pre_x_list, dist_safe = plan_enemy_first(
        board,
        obs_main,
        (shooter.x, shooter.y),
        enemies_xy,
        obs_mask_loose=obs_loose,
        shrink_list=shrink_variants
    )

    # grid -> steps
    steps_raw   = steps_from_grid_path(grid_path, pre_x_list, board, dist_safe, obs_main)
    steps_final = merge_steps(steps_raw)
    expr  = build_step_expression(steps_final)

    # overlay
    overlay = roi.copy() if overlay_path else None
    if overlay is not None:
        if obs_contours: cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)
        h,w = obs_raw.shape[:2]; cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,255,0),2)

        for a in actors:
            if a is shooter:
                cv2.circle(overlay,(a.px,a.py),18,(0,215,255),3)
                cv2.putText(overlay,"S",(a.px+10,a.py-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,215,255),2,cv2.LINE_AA)
            else:
                color=(0,0,255) if a.side=="enemy" else (255,0,0)
                label="E" if a.side=="enemy" else "A"
                cv2.circle(overlay,(a.px,a.py),6,color,2)
                cv2.putText(overlay,label,(a.px+8,a.py-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
            if a.side=="enemy":
                rpx=int((ENEMY_RADIUS/(board.x_range[1]-board.x_range[0]))*board.w)
                cv2.circle(overlay,(a.px,a.py),max(6,rpx),(0,140,255),1)

        # draw y = f(x) - f(xs) + ys
        xs = np.linspace(board.x_range[0], board.x_range[1], 1600)
        ys = eval_steps_autotranslated(xs, steps_final, xs0=shooter.x, y0=shooter.y)
        msk = (ys>=board.y_range[0]) & (ys<=board.y_range[1])
        xs,ys = xs[msk], ys[msk]
        if len(xs)>=2:
            p0 = board.xy_to_px(xs[0], ys[0])
            for k in range(1,len(xs)):
                p1 = board.xy_to_px(xs[k], ys[k])
                cv2.line(overlay, p0, p1, (0,0,255), 2)
                p0 = p1

        for (x_step, _) in steps_final:
            px,_ = board.xy_to_px(x_step, 0)
            cv2.line(overlay,(px,0),(px,board.h-1),(80,80,200),1)

    # hearts (after we guarantee enemies)
    if flex and overlay is not None:
        ally_pts = [(a.x,a.y) for a in actors if a.side=="ally"]
        heart_expr = insert_hearts(
            board, obs_main, steps_final,
            x0=shooter.x, x1=board.x_range[1], y0=shooter.y,
            overlay=overlay, ally_pts=ally_pts
        )
        if heart_expr:
            expr = (expr + " + " + heart_expr).replace("+-","-").replace("--","+").strip(" +")

    if overlay is not None:
        cv2.imwrite(overlay_path, overlay)

    out_actors=[{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
    return {"actors": out_actors, "expr": (expr if expr.strip() else "0")}

# ---------- glue ----------
def try_route_variants(board, base_masks: List[np.ndarray],
                       start_xy: Tuple[float,float], goal_xy: Tuple[float,float],
                       require_progress: bool=True) -> List[Tuple[float,float]]:
    sx,sy=start_xy; gx,gy=goal_xy
    best=[]
    for mask in base_masks:
        xs,ys,occ,cost,_ = build_occupancy_and_cost(board, mask, dx=GRID_DX, dy=GRID_DY)
        si,sj=nearest_idx(xs,ys,sx,sy); gi,gj=nearest_idx(xs,ys,gx,gy)
        si,sj=snap_to_free(occ,si,sj,6); gi,gj=snap_to_free(occ,gi,gj,6)
        seg=astar_monotone(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)
        if seg and (not require_progress or (seg[-1][0]-sx) >= GRID_DX*2):
            return seg
        seg=best_effort_toward(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)
        if seg and (seg[-1][0]-sx) > (best[-1][0]-sx if best else 0.0):
            best=seg
    return best

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Graphwar STEP-only solver (enemy-first; safe steps; hearts)")
    parser.add_argument("--xrange", nargs=2, type=float, default=[-25.0, 25.0])
    parser.add_argument("--yrange", nargs=2, type=float, default=[-15.0, 15.0])
    parser.add_argument("--min_area", type=int, default=60)
    parser.add_argument("--debug_out", default="graphwar_overlay.png")
    parser.add_argument("--flex", action="store_true", help="Insert hearts on long safe plateaus")
    args = parser.parse_args()

    bgr = capture_fullscreen_bgr()
    res = solve_from_bgr(
        bgr,
        x_range=(args.xrange[0], args.xrange[1]),
        y_range=(args.yrange[0], args.yrange[1]),
        min_area=args.min_area,
        overlay_path=(args.debug_out if args.debug_out else None),
        flex=args.flex
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
