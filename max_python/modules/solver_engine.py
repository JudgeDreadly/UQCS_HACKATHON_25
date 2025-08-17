# modules/solver_engine.py
#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

# ---- Tunables ----
ENEMY_RADIUS       = 0.8
X_GROUP_TOL        = 0.6
BORDER_THICK_PX    = 8
INFLATE_PX         = 2     # baseline thickness for raw obstacles (not the safety margin)

# NEW: pre-filter thresholds to remove tiny / thin components (e.g., text)
MIN_OBS_AREA_PX    = 150   # drop components smaller than this pixel area
MIN_OBS_THICK_PX   = 6.0   # drop components whose distance-transform thickness < this

# ---------------- Data ----------------
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

# ---------- Obstacles ----------
def _remove_axes(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                            minLineLength=int(0.5*mask.shape[1]), maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(mask, (x1,y1), (x2,y2), 0, 5)
    return mask

def _filter_small_and_thin(mask: np.ndarray,
                           min_area_px: int = MIN_OBS_AREA_PX,
                           min_thick_px: float = MIN_OBS_THICK_PX) -> np.ndarray:
    """
    Keep only components that look like real obstacles:
      - area >= min_area_px
      - OR have enough thickness by distance transform (>= min_thick_px)
    This removes text-like specks before any padding.
    """
    # Connected components
    num, labels = cv2.connectedComponents(mask)
    if num <= 1:
        return mask

    keep = np.zeros_like(mask)
    h, w = mask.shape[:2]

    for i in range(1, num):
        comp = np.uint8(labels == i) * 255
        area = int((comp > 0).sum())
        if area < min_area_px:
            # maybe still keep if it's actually thick (e.g., tiny dense chunk)
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
            thick = float(dist.max())
            if thick >= min_thick_px:
                keep = cv2.bitwise_or(keep, comp)
            continue

        # if area ok, also check thickness to avoid text strokes
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max())
        if thick >= min_thick_px:
            keep = cv2.bitwise_or(keep, comp)
        else:
            # Optional: allow “large but thin” curved blobs if quite round
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                per = cv2.arcLength(cnt, True)
                circ = 4 * math.pi * cv2.contourArea(cnt) / (per * per + 1e-9)
                if circ >= 0.60 and area >= (min_area_px * 1.5):
                    keep = cv2.bitwise_or(keep, comp)

    return keep

def obstacle_mask_and_contours(roi_bgr: np.ndarray):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    # initial “black fill” detection
    mask = np.uint8((v < 60) * 255)
    # light clean
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    # drop axes
    mask = _remove_axes(mask)
    # *** NEW: remove tiny / thin components BEFORE any padding ***
    mask = _filter_small_and_thin(mask, MIN_OBS_AREA_PX, MIN_OBS_THICK_PX)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
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

# ---------- Actors ----------
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

# ---------- Shooter ----------
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

# ---------- Public: unified solver entry ----------
def solve_from_bgr(
    bgr: np.ndarray,
    *,
    x_range: Tuple[float,float]=(-25.0,25.0),
    y_range: Tuple[float,float]=(-15.0,15.0),
    min_area: int=60,
    overlay_path: Optional[str]=None,
    solver_mode: str = "moves",       # default to moves for now
    grid_dx: float = 0.25,            # finer grid
    grid_dy: float = 0.25,
    moves_algo: str = "A*",
    flex: bool = False,               # ignored by moves
    moves_safety_px: int = 8          # BIGGER default safety ring
) -> dict:
    if solver_mode.lower() == "moves":
        # Prepare SAME components as continuous
        board, roi = find_board_roi(bgr)
        board.x_range = x_range; board.y_range = y_range

        # obstacles (now pre-filtered)
        obs_raw, obs_contours = obstacle_mask_and_contours(roi)
        border_only = add_border_mask(obs_raw.shape, px=BORDER_THICK_PX)

        # inflate ONLY obstacles first, then OR with border (baseline, still tight)
        obs_infl_main = inflate(obs_raw, INFLATE_PX)
        obs_main      = cv2.bitwise_or(obs_infl_main, border_only)

        # --- Planning safety: dilate obs+border (so planner keeps extra distance) ---
        safety_px = int(max(0, moves_safety_px))
        if safety_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*safety_px+1, 2*safety_px+1))
            obs_plan = cv2.dilate(obs_main, k, iterations=1)
        else:
            obs_plan = obs_main.copy()

        # actors & shooter
        actors = detect_actors(roi, min_area=min_area)
        assign_coords(board, actors); classify_by_center(actors)
        shooter = choose_shooter_by_red_ring(roi, actors)
        if shooter is None:
            left_allies=[a for a in actors if a.side=="ally"]
            shooter = choose_leftmost(left_allies) or choose_leftmost(actors)
        if shooter is None:
            raise RuntimeError("No actors detected.")

        # moves solver on shared components
        from modules.moves_solver import solve_moves_on_components
        res = solve_moves_on_components(
            board=board,
            roi=roi,
            obs_main=obs_main,
            obs_plan=obs_plan,
            obs_contours=obs_contours,
            actors=actors,
            shooter=shooter,
            grid_dx=grid_dx,
            grid_dy=grid_dy,
            algo=moves_algo,
            overlay_path=overlay_path
        )
        out_actors=[{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
        return {
            "actors": out_actors,
            "expr": res["expr"],
            "overlay_bgr": res["overlay_bgr"],               # ROI overlay
            "crit_overlay_bgr": res.get("crit_overlay_bgr")  # critical pixels grid
        }

    # ---------------- Minimal continuous stub (unchanged) ----------------
    board, roi = find_board_roi(bgr)
    board.x_range = x_range; board.y_range = y_range
    obs_raw, obs_contours = obstacle_mask_and_contours(roi)
    border_only = add_border_mask(obs_raw.shape, px=BORDER_THICK_PX)
    obs_infl_main = inflate(obs_raw, INFLATE_PX)
    obs_main = cv2.bitwise_or(obs_infl_main, border_only)

    actors = detect_actors(roi, min_area=min_area)
    assign_coords(board, actors); classify_by_center(actors)
    shooter = choose_leftmost([a for a in actors if a.side == "ally"]) or (actors[0] if actors else None)
    expr = "0"

    overlay = roi.copy()
    if obs_contours: cv2.drawContours(overlay, obs_contours, -1, (0,255,0), 2)
    h,w = obs_main.shape[:2]; cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,255,0),2)
    for a in actors:
        if a is shooter:
            cv2.circle(overlay,(a.px,a.py),18,(0,215,255),3)
            cv2.putText(overlay,"S",(a.px+10,a.py-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,215,255),2,cv2.LINE_AA)
        else:
            color=(0,0,255) if a.side=="enemy" else (255,0,0)
            label="E" if a.side=="enemy" else "A"
            cv2.circle(overlay,(a.px,a.py),6,color,2)
            cv2.putText(overlay,label,(a.px+8,a.py-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

    if overlay_path:
        try: cv2.imwrite(overlay_path, overlay)
        except Exception: pass

    out_actors=[{"x":float(a.x),"y":float(a.y),"side":a.side,"is_shooter":(a is shooter)} for a in actors]
    return {"actors": out_actors, "expr": expr, "overlay_bgr": overlay}
