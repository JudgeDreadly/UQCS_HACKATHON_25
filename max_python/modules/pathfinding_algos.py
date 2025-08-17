# modules/pathfinding_algos.py
from __future__ import annotations
from typing import Dict, Tuple, List
from heapq import heappush, heappop
import collections

# Allowed moves: Right, Up, Down, Diagonal Up-Right, Diagonal Down-Right
MOVES: Dict[str, Tuple[int, int]] = {
    "R":  (1,  0),
    "UR": (1, -1),
    "DR": (1,  1),
    "U":  (0, -1),
    "D":  (0,  1),
}

def _in_bounds(occ, c, r):
    return 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]

def _neighbors(occ, c, r):
    for m, (dc, dr) in MOVES.items():
        nc, nr = c + dc, r + dr
        if _in_bounds(occ, nc, nr) and occ[nr, nc] == 0:
            yield m, (nc, nr)

def _reconstruct(came, start, goal):
    if goal not in came:
        return []
    cur = goal
    out: List[Tuple[str, Tuple[int, int]]] = []
    while cur != start:
        pmove, prev = came[cur]
        out.append((pmove, cur))
        cur = prev
    out.reverse()
    return out

def _manhattan(a, b):
    (c1, r1), (c2, r2) = a, b
    return abs(c1 - c2) + abs(r1 - r2)

def astar(occ, start, goal):
    g = {start: 0}
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start))
    seen = set()
    while pq:
        _, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            ng = g[cur] + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng + _manhattan(nxt, goal), nxt))
    # fallback to best known
    if not g:
        return []
    best = min(g, key=g.get)
    return _reconstruct(came, start, best)

def dijkstra(occ, start, goal):
    g = {start: 0}
    came = {}
    pq = []
    heappush(pq, (0, start))
    seen = set()
    while pq:
        cost, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            ng = cost + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = (m, cur)
                heappush(pq, (ng, nxt))
    if not g:
        return []
    best = min(g, key=g.get)
    return _reconstruct(came, start, best)

def greedy_best_first(occ, start, goal):
    came = {}
    pq = []
    heappush(pq, (_manhattan(start, goal), start))
    seen = set()
    while pq:
        _, cur = heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            if nxt not in seen:
                came[nxt] = (m, cur)
                heappush(pq, (_manhattan(nxt, goal), nxt))
    return _reconstruct(came, start, cur)

def bfs(occ, start, goal):
    came = {}
    q = collections.deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return _reconstruct(came, start, goal)
        for m, nxt in _neighbors(occ, *cur):
            if nxt not in seen:
                seen.add(nxt)
                came[nxt] = (m, cur)
                q.append(nxt)
    return _reconstruct(came, start, cur)

PLANNERS = {
    "A*": astar,
    "Dijkstra": dijkstra,
    "Greedy": greedy_best_first,
    "BFS": bfs,
}
