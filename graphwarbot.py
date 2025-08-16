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
RDP_EPS         = 0.45     # path simplification tolerance (units)

GRID_DX         = 0.20     # A* grid step (x)
GRID_DY         = 0.20     # A* grid step (y)
PROX_WEIGHT     = 6.0      # bias to center of free space (squared)
TURN_PENALTY    = 0.20     # extra cost when changing move direction

# orientation bias (favor horizontal)
ORIENT_PENALTY_VERT = 0.20  # vertical moves slightly worse
ORIENT_BONUS_HORZ   = 0.15  # bonus for pure +x

BORDER_THICK_PX = 8         # map border as obstacle
INFLATE_PX      = 2         # safety inflation (pixels)
INFLATE_PX_LO   = 1         # looser retry inflation

# segment -> primitive
VERT_EPS_X      = 0.45      # if |dx| < this -> near-vertical (use step)
STEP_STEEPNESS  = 55.0      # 'a' for logistic step

# Flex (hearts)
HEART_REQ_W     = 10.0
HEART_REQ_H     = 10.0
HEART_GATE_A    = 80.0
HEART_GATE_D    = 0.5

BEST_EFFORT_MAX_EXPANSIONS = 120000  # cap to keep searches bounded
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

# ---------- name tags (to subtract from obstacles) ----------
def detect_white_labels(roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Find white/near-white nameboxes (with light saturation)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # allow a bit more saturation to catch tints/shadows around the label
    mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([179, 70, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w/max(1,h)
        if 150<=area<=12000 and ar>=1.05:
            boxes.append((x,y,w,h))
    return boxes

def erase_labels_from_mask(mask: np.ndarray, roi_bgr: np.ndarray, pad:int=24, dilate:int=22) -> np.ndarray:
    """Erase name tags AND their nearby dark glyphs/shadows from obstacle mask."""
    boxes = detect_white_labels(roi_bgr)
    if not boxes: return mask
    lab = np.zeros_like(mask)
    for (x,y,w,h) in boxes:
        x0=max(0,x-pad); y0=max(0,y-pad)
        x1=min(mask.shape[1]-1,x+w+pad); y1=min(mask.shape[0]-1,y+h+pad)
        cv2.rectangle(lab,(x0,y0),(x1,y1),255,thickness=cv2.FILLED)
    if dilate>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate+1,2*dilate+1))
        lab=cv2.dilate(lab,k,iterations=1)
    out=mask.copy()
    out[lab>0]=0
    return out

# ---------- obstacles ----------
def _remove_axes(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=int(0.5*mask.shape[1]), maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(mask, (x1,y1), (x2,y2), 0, 5)
    return mask

def obstacle_mask_and_contours(roi_bgr: np.ndarray):
    """Dark obstacles (circles/blobs); axes removed; labels erased."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)
    mask = _remove_axes(mask)
    mask = erase_labels_from_mask(mask, roi_bgr, pad=24, dilate=22)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c)>=200]
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, kept, -1, 255, thickness=cv2.FILLED)
    return clean, kept

def add_border_as_obstacle(mask: np.ndarray, px:int=BORDER_THICK_PX)->np.ndarray:
    h,w = mask.shape[:2]; b = np.zeros_like(mask)
    cv2.rectangle(b,(0,0),(w-1,h-1),255,thickness=px)
    return cv2.bitwise_or(mask,b)

def inflate_for_safety(mask: np.ndarray, px:int=INFLATE_PX)->np.ndarray:
    if px<=0: return mask.copy()
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(px,px))
    return cv2.dilate(mask,k,iterations=1)

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

def filter_out_name_badges(roi_bgr: np.ndarray, actors: List[Actor])->List[Actor]:
    boxes = detect_white_labels(roi_bgr); keep=[]
    for a in actors:
        # ditch tiny color specks sitting on top of a label
        near=False
        for (x,y,w,h) in boxes:
            cx,cy=x+w//2,y+h//2
            if (a.px-cx)**2+(a.py-cy)**2 <= (36**2): near=True; break
        if near and a.area<220: continue
        keep.append(a)
    return keep

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

# ---------- bridges ----------
def diagonal_params_from_line(xs: float, xe: float, slope_m: float)->Tuple[float,float,float]:
    start,end = (xs,xe) if xs<xe else (xe,xs)
    a = abs(slope_m)/2.0
    if slope_m>=0: b,c = -start,-end
    else:          b,c = -end,-start
    return a,b,c

def segment_to_bridge(x1: float, y1: float, x2: float, y2: float) -> Tuple[str, Dict]:
    dx = x2 - x1; dy = y2 - y1
    if abs(dx) >= VERT_EPS_X:
        m = dy/dx if dx!=0 else 0.0
        a,b,c = diagonal_params_from_line(x1, x2, m)
        term = f"{a:.4f}*(abs(x+({b:+.4f}))-abs(x+({c:+.4f})))"
        return term, {"type":"diagonal","a":a,"b":b,"c":c,"start":x1,"end":x2,"slope":m}
    # near-vertical -> STEP (logistic)
    k = dy
    a = STEP_STEEPNESS
    c = -x1
    term = f"{k:.4f}/(1+exp(-{a:.0f}*(x+({c:+.4f}))))"
    return term, {"type":"step","k":k,"a":a,"c":c,"x_at":x1}

def build_expression_from_polyline(pts: List[Tuple[float,float]]) -> Tuple[str, List[Dict]]:
    if len(pts)<2: return "0", []
    parts=[]; meta=[]
    for i in range(1,len(pts)):
        (x1,y1),(x2,y2) = pts[i-1], pts[i]
        if abs(x2-x1)<1e-9 and abs(y2-y1)<1e-9: continue
        term,info = segment_to_bridge(x1,y1,x2,y2)
        parts.append(term); meta.append(info)
    expr = " + ".join(parts)
    return expr.replace("+-","-").replace("--","+"), meta

# ---------- RDP with anchors ----------
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

def rdp_with_anchors(points: List[Tuple[float,float]], anchor_idx: List[int], eps: float)->List[Tuple[float,float]]:
    anchor_idx=sorted(set([max(0,min(len(points)-1,i)) for i in anchor_idx]))
    if not anchor_idx: return rdp_segment(points,eps)
    out=[]
    for a,b in zip(anchor_idx[:-1], anchor_idx[1:]):
        seg=points[a:b+1]; simp=rdp_segment(seg,eps)
        if out: out.extend(simp[1:])
        else:   out.extend(simp)
    return out

# ---------- grid / A* ----------
def build_occupancy_and_cost(board: Board, obs_mask: np.ndarray,
                             dx: float=GRID_DX, dy: float=GRID_DY, w_prox: float=PROX_WEIGHT):
    xs = np.arange(board.x_range[0], board.x_range[1]+1e-9, dx)
    ys = np.arange(board.y_range[0], board.y_range[1]+1e-9, dy)
    free=(obs_mask==0).astype(np.uint8)*255
    dist=cv2.distanceTransform(free, cv2.DIST_L2, 5)
    dist_norm=dist/dist.max() if dist.max()>0 else dist
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
    """Monotone A*: advances in +x; prefers horizontal; can drop early."""
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
    """A*-like exploration without success criterion; returns path to the visited node
       with the smallest f = g + h (closest reachable cell toward goal)."""
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
    # reconstruct to best_node
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

# ---------- plan path ----------
def plan_path(board: Board,
              obs_mask_main: np.ndarray,
              soldier_xy: Tuple[float,float],
              enemies_xy: List[Tuple[float,float]],
              obs_mask_loose: Optional[np.ndarray]=None
              ) -> Tuple[List[Tuple[float,float]], List[int], np.ndarray]:
    xs,ys,occ,cost,dist = build_occupancy_and_cost(board, obs_mask_main, dx=GRID_DX, dy=GRID_DY, w_prox=PROX_WEIGHT)
    if obs_mask_loose is not None:
        xsL,ysL,occL,costL,_ = build_occupancy_and_cost(board, obs_mask_loose, dx=GRID_DX, dy=GRID_DY, w_prox=PROX_WEIGHT)
    else:
        xsL=ysL=occL=costL=None

    sx,sy=soldier_xy
    cols=group_enemies_by_x(enemies_xy,start_x=sx,x_tol=X_GROUP_TOL)

    path=[(sx,sy)]; anchors=[0]; curx,cury=sx,sy
    for col in cols:
        xg=max(p[0] for p in col)             # column x (rightmost in the group)
        ig,_=nearest_idx(xs,ys,xg,cury)       # main-grid i for target column
        remaining=col[:]
        reached_any=False
        while remaining:
            remaining.sort(key=lambda p: abs(p[1]-cury))
            tx,ty=remaining.pop(0)

            # --- 1) main exact ---
            si,sj=nearest_idx(xs,ys,curx,cury)
            gi,gj=ig, nearest_idx(xs,ys,xg,ty)[1]
            si,sj=snap_to_free(occ,si,sj,max_r=6); gi,gj=snap_to_free(occ,gi,gj,max_r=6)
            seg=astar_monotone(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)

            # --- 2) loose exact ---
            if not seg and xsL is not None:
                siL,sjL=nearest_idx(xsL,ysL,curx,cury)
                giL,gjL=nearest_idx(xsL,ysL,xg,ty)
                siL,sjL=snap_to_free(occL,siL,sjL,max_r=6); giL,gjL=snap_to_free(occL,giL,gjL,max_r=6)
                seg=astar_monotone(xsL,ysL,occL,costL,(siL,sjL),(giL,gjL),dir_sign=+1)

            # --- 3) main best-effort ---
            if not seg:
                seg=best_effort_toward(xs,ys,occ,cost,(si,sj),(gi,gj),dir_sign=+1)

            # --- 4) loose best-effort ---
            if not seg and xsL is not None:
                seg=best_effort_toward(xsL,ysL,occL,costL,(siL,sjL),(giL,gjL),dir_sign=+1)

            if seg:
                if path: seg=seg[1:]
                path.extend(seg); anchors.append(len(path)-1)
                curx,cury=path[-1]
                reached_any=True
            # else: unreachable even best-effort; try next target in the same column

        # if entire column unreachable, we skip it and continue to the next column
    return path,anchors,dist

# ---------- flex hearts ----------
def heart_term(a: float, L: float, R: float, scale: float) -> str:
    h=(f"0.4*((abs(x-({a:.4f}))-1.5)-abs(abs(x-({a:.4f}))-1.5))"
       f"+sqrt(2.25-(1.5+0.4*((abs(x-({a:.4f}))-1.5)-abs(abs(x-({a:.4f}))-1.5)))^2)*cos(30*x)")
    gate=(f"(1/(1+exp(-{HEART_GATE_A:.1f}*(x-({L+HEART_GATE_D:.4f}))))"
          f"-1/(1+exp(-{HEART_GATE_A:.1f}*(x-({R-HEART_GATE_D:.4f})))))")
    return f"{scale:.4f}*({h})*{gate}"

def insert_flex_hearts(board: Board, dist_map: np.ndarray, path_simple: List[Tuple[float,float]],
                       overlay: Optional[np.ndarray])->str:
    if len(path_simple)<2: return ""
    exprs=[]
    units_per_px_y=(board.y_range[1]-board.y_range[0])/board.h
    for i in range(1,len(path_simple)):
        (x1,y1),(x2,y2)=path_simple[i-1], path_simple[i]
        if x2<=x1: continue
        dx=x2-x1
        if dx<HEART_REQ_W: continue
        L=x1+(dx-HEART_REQ_W)/2.0; R=L+HEART_REQ_W
        samples=40; min_clear_units=1e9
        for t in np.linspace(0,1,samples):
            x=L+(R-L)*t; y=y1 + (y2-y1)*((x-x1)/(x2-x1))
            px,py=board.xy_to_px(x,y)
            px=int(np.clip(px,0,board.w-1)); py=int(np.clip(py,0,board.h-1))
            d_units=float(dist_map[py,px])*units_per_px_y
            min_clear_units=min(min_clear_units,d_units)
        margin=0.8; usable=max(0.0, min_clear_units-margin)
        if usable <= HEART_REQ_H/6.0:
            continue
        scale=min(1.0, usable/3.0)
        a=L; exprs.append(heart_term(a,L,R,scale))
        if overlay is not None:
            pL=board.xy_to_px(L,y1+(y2-y1)*((L-x1)/(x2-x1)))
            pR=board.xy_to_px(R,y1+(y2-y1)*((R-x1)/(x2-x1)))
            cv2.line(overlay,(pL[0],0),(pL[0],board.h-1),(200,0,200),1)
            cv2.line(overlay,(pR[0],0),(pR[0],board.h-1),(200,0,200),1)
    return " + ".join(exprs)

# ---------- solve ----------
def solve_from_bgr(bgr: np.ndarray,
                   x_range: Tuple[float,float]=(-25.0,25.0), y_range: Tuple[float,float]=(-15.0,15.0),
                   min_area: int=60, overlay_path: Optional[str]=None, flex: bool=False)->dict:
    if cv2 is None: raise RuntimeError("Requires OpenCV.")
    board,roi = find_board_roi(bgr); board.x_range=x_range; board.y_range=y_range
    # obstacles
    obs_raw,obs_contours = obstacle_mask_and_contours(roi)
    obs_bord = add_border_as_obstacle(obs_raw, px=BORDER_THICK_PX)
    obs_main = inflate_for_safety(obs_bord, px=INFLATE_PX)
    obs_loose= inflate_for_safety(obs_bord, px=INFLATE_PX_LO)

    # actors
    actors_all = detect_actors(roi, min_area=min_area)
    actors     = filter_out_name_badges(roi, actors_all)
    assign_coords(board, actors); classify_by_center(actors)

    # shooter
    shooter = choose_shooter_by_red_ring(roi, actors)
    if shooter is None:
        left_allies=[a for a in actors if a.side=="ally"]
        shooter = choose_leftmost(left_allies) or choose_leftmost(actors)
    if shooter is None: raise RuntimeError("No actors detected.")

    enemies_xy=[(a.x,a.y) for a in actors if a.side=="enemy"]

    # plan (with loose & best-effort fallbacks)
    raw_path, anchors, dist_map = plan_path(board, obs_main, (shooter.x, shooter.y),
                                            enemies_xy, obs_mask_loose=obs_loose)
    path = rdp_with_anchors(raw_path, anchors, eps=RDP_EPS)
    base_expr, _ = build_expression_from_polyline(path)

    # overlay
    overlay = roi.copy() if overlay_path else None
    if overlay is not None:
        if obs_contours: cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)
        h,w = obs_bord.shape[:2]; cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,255,0),2)
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
        if len(path)>=2:
            pts=[board.xy_to_px(x,y) for (x,y) in path]
            for i in range(1,len(pts)):
                cv2.line(overlay, pts[i-1], pts[i], (0,0,255), 2)

    # flex
    flex_expr=""
    if flex:
        flex_expr = insert_flex_hearts(board, dist_map, path, overlay)
    final_expr = base_expr if not flex_expr else (base_expr + " + " + flex_expr).replace("+-","-").replace("--","+").strip(" +")

    if overlay is not None:
        cv2.imwrite(overlay_path, overlay)

    out_actors=[{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
    return {"actors": out_actors, "expr": final_expr if final_expr.strip() else "0"}

# ---------- cli ----------
def main():
    parser = argparse.ArgumentParser(description="Graphwar solver (name-tag fix + best-effort)")
    parser.add_argument("--xrange", nargs=2, type=float, default=[-25.0, 25.0])
    parser.add_argument("--yrange", nargs=2, type=float, default=[-15.0, 15.0])
    parser.add_argument("--min_area", type=int, default=60)
    parser.add_argument("--debug_out", default="graphwar_overlay.png")
    parser.add_argument("--flex", action="store_true", help="Insert decorative hearts where safe")
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
