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

# -------------------- tunables --------------------
ENEMY_RADIUS    = 0.8      # units; leeway ring for enemy hits (visual only)
X_GROUP_TOL     = 0.6      # group enemies by x within this tolerance

# Grid & path cost
GRID_DX         = 0.20
GRID_DY         = 0.20
PROX_WEIGHT     = 6.0
TURN_PENALTY    = 0.20
ORIENT_PENALTY_VERT = 0.20
ORIENT_BONUS_HORZ   = 0.15

# Map border / obstacle shaping
BORDER_THICK_PX = 8
INFLATE_PX      = 2
INFLATE_PX_LO   = 1
SHRINK_STEPS    = 6
MIN_THICK_PX    = 6.0

# --- STEP-ONLY settings ---
STEP_STEEPNESS       = 69.0   # 'a' in k/(1+exp(-a*(x-c)))
STEP_EPS_DY          = 0.35   # ignore tiny vertical changes
STEP_OFFSET_FIRST    = 0.25   # first step must not be at shooter's x
STEP_OFFSET          = 0.12   # default offset for later steps
STEP_MAX_OFFSET_FRAC = 0.33   # at most 33% into the next span
STEP_MIN_GAP_AFTER   = 0.05   # keep step a bit before the next vertex

# Smoothing of geometric path (for nicer long straights)
SMOOTH_ITERS    = 3
SMOOTH_STEP_PX  = 1.5
SMOOTH_MAX_DY   = 2.0

BEST_EFFORT_MAX_EXPANSIONS = 120000

# “must-cross-enemy” x offsets (units)
X_PRE_EPS       = 0.35
X_POST_EPS      = 0.25
# --------------------------------------------------

# ---------- capture ----------
def capture_fullscreen_bgr() -> np.ndarray:
    try:
        import mss
    except Exception as e:
        raise RuntimeError("Screenshot capture requires: pip install mss") from e
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])  # primary monitor
        frame = np.array(sct_img)            # BGRA
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

# ---------- ROI / board ----------
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

# ---------- label detection & thin-stroke removal ----------
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
        if area < 80:
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
    mask = np.uint8((v < 60) * 255)  # true black fill only
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

# ---------- aura shooter ----------
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

# ---------- STEP-ONLY expression ----------
def steps_from_path(points: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    """
    Convert a monotone-x path into a sequence of steps.
    Each step: (x_step, k), where k is the vertical jump applied at x_step.
    """
    steps: List[Tuple[float,float]] = []
    if len(points) < 2:
        return steps

    for i in range(len(points)-1):
        x1,y1 = points[i]
        x2,y2 = points[i+1]
        if x2 <= x1 + 1e-6:
            continue  # ignore non-forward moves
        dy = y2 - y1
        if abs(dy) < STEP_EPS_DY:
            continue  # keep long straights: no step if almost flat

        dx = x2 - x1
        # choose a step location within (x1, x2]
        if i == 0:
            base_off = STEP_OFFSET_FIRST
        else:
            base_off = STEP_OFFSET
        max_off = max(base_off, min(STEP_MAX_OFFSET_FRAC*dx, dx - STEP_MIN_GAP_AFTER))
        x_step = x1 + max_off
        steps.append((x_step, dy))
    return steps

def build_step_expression(steps: List[Tuple[float,float]]) -> str:
    if not steps:
        return "0"
    parts=[]
    for (x_step,k) in steps:
        # k/(1+exp(-a*(x - x_step)))  -> using (x + c) form with c=-x_step
        c = -x_step
        parts.append(f"{k:.4f}/(1+exp(-{STEP_STEEPNESS:.0f}*(x+({c:+.4f}))))")
    expr = " + ".join(parts)
    return expr.replace("+-","-").replace("--","+")

def eval_steps(xs: np.ndarray, steps: List[Tuple[float,float]], y0: float) -> np.ndarray:
    y = np.full_like(xs, y0, dtype=float)
    if not steps:
        return y
    for (x_step,k) in steps:
        y += k / (1.0 + np.exp(-STEP_STEEPNESS*(xs - x_step)))
    return y

# ---------- RDP ----------
def rdp_segment(points: List[Tuple[float,float]], eps: float)->List[Tuple[float,float]]:
    if len(points)<3: return points[:]
    def dist(p,a,b):
        (x,y),(x1,y1),(x2,y2) = p,a,b
        if x1==x2 and y1==y2: return math.hypot(x-x1, y-y1)
        t=((x-x1)*(x2-x1)+(y-y1)*(y2-y1))/((x2-x1)**2+(y2-y1)**2)
        t=max(0.0,min(1.0,t))
        xp,yp = x1+t*(x2-x1), y1+t*(y2-y1)
        return math.hypot(x-xp,y-yp)
    a,b = points[0],points[-1]
    idx,dmax=0,0.0
    for i in range(1,len(points)-1):
        d=dist(points[i],a,b)
        if d>dmax: idx,dmax=i,d
    if dmax>eps:
        left=rdp_segment(points[:idx+1],eps)
        right=rdp_segment(points[idx:],eps)
        return left[:-1]+right
    return [a,b]

# ---------- grid / A* ----------
def build_occupancy_and_cost(board: Board, obs_mask: np.ndarray,
                             dx: float=GRID_DX, dy: float=GRID_DY, w_prox: float=PROX_WEIGHT):
    xs = np.arange(board.x_range[0], board.x_range[1]+1e-9, dx)
    ys = np.arange(board.y_range[0], board.y_range[1]+1e-9, dy)
    free=(obs_mask==0).astype(np.uint8)*255
    dist=cv2.distanceTransform(free, cv2.DIST_L2, 5)
    dist_norm=dist/dist.max() if dist.max() > 0 else dist
    occ=np.zeros((len(ys),len(xs)),dtype=np.uint8)
    cost=np.zeros_like(occ,dtype=np.float32)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            px,py=board.xy_to_px(x,y)
            px=int(np.clip(px,0,board.w-1)); py=int(np.clip(py,0,board.h-1))
            occ[j,i] = 1 if obs_mask[py,px]>0 else 0
            prox = 1.0 - float(dist_norm[py,px])
            cost[j,i]=1.0 + w_prox*(prox**2)
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
    moves=[(dir_sign,0),(dir_sign,+1),(dir_sign,-1),(0,+1),(0,-1)]
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
    moves=[(dir_sign,0),(dir_sign,+1),(dir_sign,-1),(0,+1),(0,-1)]
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

# ---------- enemy grouping ----------
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

# ---------- smoothing ----------
def smooth_path_away_from_obstacles(board: Board,
                                    path: List[Tuple[float,float]],
                                    dist_map: np.ndarray,
                                    obs_mask: np.ndarray) -> List[Tuple[float,float]]:
    if len(path)<3: return path[:]
    smoothed = path[:]
    dm = cv2.GaussianBlur(dist_map.astype(np.float32), (0,0), 1.0)
    gy, gx = np.gradient(dm)
    units_per_px_x = (board.x_range[1]-board.x_range[0])/board.w
    units_per_px_y = (board.y_range[1]-board.y_range[0])/board.h
    for _ in range(SMOOTH_ITERS):
        for i in range(1, len(smoothed)-1):
            x,y = smoothed[i]
            px,py = board.xy_to_px(x,y)
            px=np.clip(px,0,board.w-1); py=np.clip(py,0,board.h-1)
            gxv = float(gx[py,px]); gyv = float(gy[py,px])
            mag = math.hypot(gxv, gyv)
            if mag < 1e-6: continue
            dx_px = (gxv/mag)*SMOOTH_STEP_PX
            dy_px = -(gyv/mag)*SMOOTH_STEP_PX
            dx = dx_px*units_per_px_x
            dy = dy_px*units_per_px_y
            nx = max(smoothed[i-1][0]+1e-4, min(smoothed[i+1][0]-1e-4, x + max(0.0, dx)))
            ny = y + np.clip(dy, -SMOOTH_MAX_DY, SMOOTH_MAX_DY)
            npx,npy = board.xy_to_px(nx,ny)
            npx=np.clip(npx,0,board.w-1); npy=np.clip(npy,0,board.h-1)
            if obs_mask[npy,npx]==0:
                smoothed[i]=(nx,ny)
    return smoothed

# ---------- plan path ----------
def try_route_variants(board, base_masks: List[np.ndarray],
                       start_xy: Tuple[float,float], goal_xy: Tuple[float,float],
                       require_progress: bool=True) -> List[Tuple[float,float]]:
    sx,sy=start_xy; gx,gy=goal_xy
    best=[]
    for mask in base_masks:
        xs,ys,occ,cost,_ = build_occupancy_and_cost(board, mask, dx=GRID_DX, dy=GRID_DY, w_prox=PROX_WEIGHT)
        si,sj=nearest_idx(xs,ys,sx,sy); gi,gj=nearest_idx(xs,ys,gx,gy)
        si,sj=snap_to_free(occ,si,sj,6); gi,gj=snap_to_free(occ,gi,gj,6)
        seg=astar_monotone(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)
        if seg and (not require_progress or (seg[-1][0]-sx) >= GRID_DX*2):
            return seg
        seg=best_effort_toward(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)
        if seg and (seg[-1][0]-sx) > (best[-1][0]-sx if best else 0.0):
            best=seg
    return best

def plan_path(board: Board,
              obs_mask_main: np.ndarray,
              soldier_xy: Tuple[float,float],
              enemies_xy: List[Tuple[float,float]],
              obs_mask_loose: Optional[np.ndarray]=None,
              shrink_list: Optional[List[np.ndarray]]=None
              ) -> Tuple[List[Tuple[float,float]], List[int], np.ndarray, np.ndarray]:
    xs0,ys0,occ0,cost0,dist_safe = build_occupancy_and_cost(board, obs_mask_main, dx=GRID_DX, dy=GRID_DY, w_prox=PROX_WEIGHT)

    sx,sy=soldier_xy
    cols=group_enemies_by_x(enemies_xy,start_x=sx,x_tol=X_GROUP_TOL)

    path=[(sx,sy)]; anchors=[0]; curx,cury=sx,sy
    for col in cols:
        xcol = float(np.mean([x for x,_ in col]))
        x_pre  = xcol - X_PRE_EPS
        x_post = xcol + X_POST_EPS

        remaining = col[:]
        while remaining:
            remaining.sort(key=lambda p: abs(p[1]-cury))
            ex,ey = remaining.pop(0)

            goal1 = (x_pre, ey)
            masks = [obs_mask_main]
            if obs_mask_loose is not None: masks.append(obs_mask_loose)
            if shrink_list: masks.extend(shrink_list)
            seg1 = try_route_variants(board, masks, (curx,cury), goal1, require_progress=True)

            if seg1:
                if path: seg1 = seg1[1:]
                path.extend(seg1); anchors.append(len(path)-1)
                curx,cury = path[-1]
            else:
                continue

            goal2 = (x_post, cury)
            seg2 = try_route_variants(board, masks, (curx,cury), goal2, require_progress=False)
            if seg2:
                if path: seg2 = seg2[1:]
                path.extend(seg2); anchors.append(len(path)-1)
                curx,cury = path[-1]

    return path,anchors,dist_safe,obs_mask_main

# ---------- solve ----------
def solve_from_bgr(bgr: np.ndarray,
                   x_range: Tuple[float,float]=(-25.0,25.0), y_range: Tuple[float,float]=(-15.0,15.0),
                   min_area: int=60, overlay_path: Optional[str]=None)->dict:
    if cv2 is None: raise RuntimeError("Requires OpenCV.")
    board,roi = find_board_roi(bgr); board.x_range=x_range; board.y_range=y_range

    # obstacles
    obs_raw,obs_contours = obstacle_mask_and_contours(roi)

    # border (never inflated)
    border_only = add_border_mask(obs_raw.shape, px=BORDER_THICK_PX)

    # inflate ONLY obstacles, then OR with border
    obs_infl_main  = inflate(obs_raw, INFLATE_PX)
    obs_infl_loose = inflate(obs_raw, INFLATE_PX_LO)
    obs_main  = cv2.bitwise_or(obs_infl_main,  border_only)
    obs_loose = cv2.bitwise_or(obs_infl_loose, border_only)

    # progressive shrink variants (obstacles only), then add border
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

    # plan path (geometric)
    raw_path, anchors, dist_safe, _ = plan_path(
        board,
        obs_main,
        (shooter.x, shooter.y),
        enemies_xy,
        obs_mask_loose=obs_loose,
        shrink_list=shrink_variants
    )

    # simplify a bit (keeps long straights)
    tight_path = rdp_segment(raw_path, eps=0.45)
    smooth_path = smooth_path_away_from_obstacles(board, tight_path, dist_safe, obs_main)

    # ---- STEP-ONLY conversion ----
    steps = steps_from_path(smooth_path)
    expr  = build_step_expression(steps)

    # overlay (draw what the *function* will do)
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

        # draw the predicted step-curve
        xs = np.linspace(board.x_range[0], board.x_range[1], 1600)
        ys = eval_steps(xs, steps, y0=shooter.y)
        # keep inside board
        msk = (ys>=board.y_range[0]) & (ys<=board.y_range[1])
        xs,ys = xs[msk], ys[msk]
        if len(xs)>=2:
            p0 = board.xy_to_px(xs[0], ys[0])
            for k in range(1,len(xs)):
                p1 = board.xy_to_px(xs[k], ys[k])
                cv2.line(overlay, p0, p1, (0,0,255), 2)
                p0 = p1

        # draw step centers for clarity
        for (x_step, k) in steps:
            px,py = board.xy_to_px(x_step, shooter.y)
            cv2.line(overlay,(px,0),(px,board.h-1),(50,50,255),1)

        cv2.imwrite(overlay_path, overlay)

    out_actors=[{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
    return {"actors": out_actors, "expr": (expr if expr.strip() else "0")}

# ---------- cli ----------
def main():
    parser = argparse.ArgumentParser(description="Graphwar solver — STEP ONLY (prioritize long straights)")
    parser.add_argument("--xrange", nargs=2, type=float, default=[-25.0, 25.0])
    parser.add_argument("--yrange", nargs=2, type=float, default=[-15.0, 15.0])
    parser.add_argument("--min_area", type=int, default=60)
    parser.add_argument("--debug_out", default="graphwar_overlay.png")
    args = parser.parse_args()

    bgr = capture_fullscreen_bgr()
    res = solve_from_bgr(
        bgr,
        x_range=(args.xrange[0], args.xrange[1]),
        y_range=(args.yrange[0], args.yrange[1]),
        min_area=args.min_area,
        overlay_path=(args.debug_out if args.debug_out else None),
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()