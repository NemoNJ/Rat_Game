"""
search.py
=========

Path-planning module implementing four classic search algorithms over the
maze graph:

    1. Breadth-First Search (BFS)
    2. Depth-First Search (DFS)
    3. Dijkstra's Algorithm
    4. A* Search

All four algorithms share a common `SearchResult` return type (path,
path length, explored-node count/order, and computation time) so they
can be benchmarked and compared uniformly by main.py.

Every algorithm is built to run well within the competition's 3-minute
per-move "thinking time" limit: the maze graph has at most rows*cols
(<= 900) nodes and at most ~2*rows*cols edges, so even the least
efficient of the four (DFS) is bounded by O(V + E) time -- microseconds
on modern hardware -- with enormous headroom under 180 seconds.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from heuristic import manhattan_distance
from maze import Coord, Maze


@dataclass
class SearchResult:
    """
    Uniform result container returned by every search algorithm.

    Attributes:
        algorithm: Name of the algorithm that produced this result.
        path: Ordered list of coordinates from start to goal inclusive.
            Empty list if no path was found.
        path_length: Number of moves in the path (len(path) - 1), i.e.
            the number of edges traversed. 0 if no path found or start
            equals goal.
        explored_order: Coordinates in the exact order they were popped
            from the frontier / visited (useful for visualization and
            for counting total nodes explored).
        explored_count: Total number of distinct nodes explored
            (len(explored_order) as a distinct-node count).
        computation_time_s: Wall-clock time taken by the search, in
            seconds.
        success: True if a path from start to goal was found.
    """

    algorithm: str
    path: List[Coord] = field(default_factory=list)
    path_length: int = 0
    explored_order: List[Coord] = field(default_factory=list)
    explored_count: int = 0
    computation_time_s: float = 0.0
    success: bool = False

    def summary(self) -> str:
        """Return a short human-readable one-line summary of the result."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{self.algorithm:10s}] {status:7s} | "
            f"path_len={self.path_length:4d} | "
            f"explored={self.explored_count:4d} | "
            f"time={self.computation_time_s * 1000:8.3f} ms"
        )


def _reconstruct_path(
    came_from: Dict[Coord, Optional[Coord]], start: Coord, goal: Coord
) -> List[Coord]:
    """
    Walk the came_from map backwards from goal to start and return the
    forward-ordered path.

    Args:
        came_from: Mapping of each visited coordinate to its predecessor
            (start maps to None).
        start: The start coordinate.
        goal: The goal coordinate.

    Returns:
        List of coordinates from start to goal inclusive. Empty list if
        goal was never reached (not present in came_from).
    """
    if goal not in came_from:
        return []
    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


class MazeSolver:
    """
    Encapsulates all four path-planning algorithms over a given Maze.

    Each public `solve_*` method times itself, explores the maze graph
    (defined by `Maze.accessible_neighbors`, i.e. edges = removed walls),
    and returns a SearchResult. Keeping all algorithms as methods on one
    class avoids duplicating the timing/result-building boilerplate.
    """

    def __init__(self, maze: Maze) -> None:
        """
        Args:
            maze: The Maze instance to search over.
        """
        self.maze = maze

    # ------------------------------------------------------------------ #
    # 1. Breadth-First Search
    # ------------------------------------------------------------------ #

    def solve_bfs(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> SearchResult:
        """
        Breadth-First Search.

        Explores the maze graph level-by-level using a FIFO queue.
        Because every edge has identical unit cost (one grid step),
        BFS is guaranteed to return a *shortest* path in terms of number
        of moves -- identical in length to Dijkstra's result on this
        uniform-cost graph.

        Time complexity: O(V + E)
        Space complexity: O(V)

        Args:
            start: Start coordinate (default: maze.start).
            goal: Goal coordinate (default: maze.finish).

        Returns:
            SearchResult with the shortest path (by move count).
        """
        start = start if start is not None else self.maze.start
        goal = goal if goal is not None else self.maze.finish

        t0 = time.perf_counter()

        frontier: deque = deque([start])
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        explored_order: List[Coord] = []

        while frontier:
            current = frontier.popleft()
            explored_order.append(current)
            if current == goal:
                break
            for nxt in self.maze.accessible_neighbors(current):
                if nxt not in came_from:
                    came_from[nxt] = current
                    frontier.append(nxt)

        elapsed = time.perf_counter() - t0
        path = _reconstruct_path(came_from, start, goal)

        return SearchResult(
            algorithm="BFS",
            path=path,
            path_length=max(len(path) - 1, 0),
            explored_order=explored_order,
            explored_count=len(explored_order),
            computation_time_s=elapsed,
            success=bool(path),
        )

    # ------------------------------------------------------------------ #
    # 2. Depth-First Search
    # ------------------------------------------------------------------ #

    def solve_dfs(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> SearchResult:
        """
        Depth-First Search (iterative, explicit stack).

        Explores as deep as possible along each branch before
        backtracking. Unlike BFS/Dijkstra/A*, DFS does **not** guarantee
        a shortest path -- it is included because the specification
        explicitly requests it, and because it is a useful contrast case
        when comparing algorithms (typically far more path length /
        explored-node variance).

        Time complexity: O(V + E)
        Space complexity: O(V)

        Args:
            start: Start coordinate (default: maze.start).
            goal: Goal coordinate (default: maze.finish).

        Returns:
            SearchResult with *a* valid path (not necessarily shortest).
        """
        start = start if start is not None else self.maze.start
        goal = goal if goal is not None else self.maze.finish

        t0 = time.perf_counter()

        stack: List[Coord] = [start]
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        explored_order: List[Coord] = []
        visited = {start}

        while stack:
            current = stack.pop()
            explored_order.append(current)
            if current == goal:
                break
            for nxt in self.maze.accessible_neighbors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = current
                    stack.append(nxt)

        elapsed = time.perf_counter() - t0
        path = _reconstruct_path(came_from, start, goal)

        return SearchResult(
            algorithm="DFS",
            path=path,
            path_length=max(len(path) - 1, 0),
            explored_order=explored_order,
            explored_count=len(explored_order),
            computation_time_s=elapsed,
            success=bool(path),
        )

    # ------------------------------------------------------------------ #
    # 3. Dijkstra's Algorithm
    # ------------------------------------------------------------------ #

    def solve_dijkstra(self, start: Optional[Coord] = None, goal: Optional[Coord] = None) -> SearchResult:
        """
        Dijkstra's Algorithm using a binary heap priority queue.

        Every edge in this maze graph has identical unit cost, so
        Dijkstra here is mathematically equivalent to BFS in the path it
        returns; it is included (a) because the specification explicitly
        requires it, and (b) because it demonstrates the general
        weighted-shortest-path approach, which would matter immediately
        if cell costs were ever made non-uniform (e.g. rough terrain,
        diagonal cost, sensor uncertainty).

        Time complexity: O((V + E) log V) with a binary heap
        Space complexity: O(V)

        Args:
            start: Start coordinate (default: maze.start).
            goal: Goal coordinate (default: maze.finish).

        Returns:
            SearchResult with the shortest path (by total edge cost).
        """
        start = start if start is not None else self.maze.start
        goal = goal if goal is not None else self.maze.finish

        t0 = time.perf_counter()

        dist: Dict[Coord, float] = {start: 0.0}
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        explored_order: List[Coord] = []
        visited: set = set()

        # Priority queue entries: (distance, coord)
        heap: List[tuple] = [(0.0, start)]

        while heap:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue  # stale entry (superseded by a shorter path found later)
            visited.add(current)
            explored_order.append(current)

            if current == goal:
                break

            for nxt in self.maze.accessible_neighbors(current):
                if nxt in visited:
                    continue
                new_dist = current_dist + 1.0  # unit edge cost (one grid step)
                if nxt not in dist or new_dist < dist[nxt]:
                    dist[nxt] = new_dist
                    came_from[nxt] = current
                    heapq.heappush(heap, (new_dist, nxt))

        elapsed = time.perf_counter() - t0
        path = _reconstruct_path(came_from, start, goal)

        return SearchResult(
            algorithm="Dijkstra",
            path=path,
            path_length=max(len(path) - 1, 0),
            explored_order=explored_order,
            explored_count=len(explored_order),
            computation_time_s=elapsed,
            success=bool(path),
        )

    # ------------------------------------------------------------------ #
    # 4. A* Search
    # ------------------------------------------------------------------ #

    def solve_astar(
        self,
        start: Optional[Coord] = None,
        goal: Optional[Coord] = None,
        heuristic_fn: Optional[Callable[[Coord, Coord], float]] = None,
    ) -> SearchResult:
        """
        A* Search using Manhattan distance as the default heuristic.

        A* orders the frontier by f(n) = g(n) + h(n), where g(n) is the
        actual cost from start to n and h(n) is the estimated remaining
        cost to goal. Manhattan distance is admissible and consistent
        for this 4-directional unit-cost grid (it never overestimates
        true remaining distance, since walls can only make the real path
        longer, never shorter than a straight-line grid distance) --
        this guarantees A* returns an optimal (shortest) path while
        typically exploring far fewer nodes than BFS/Dijkstra by
        prioritizing cells that look closer to the goal.

        Time complexity: O((V + E) log V) with a binary heap
        Space complexity: O(V)

        Args:
            start: Start coordinate (default: maze.start).
            goal: Goal coordinate (default: maze.finish).
            heuristic_fn: Optional custom heuristic h(coord, goal) ->
                estimated cost. Defaults to Manhattan distance.

        Returns:
            SearchResult with the optimal (shortest) path.
        """
        start = start if start is not None else self.maze.start
        goal = goal if goal is not None else self.maze.finish
        heuristic_fn = heuristic_fn if heuristic_fn is not None else manhattan_distance

        t0 = time.perf_counter()

        g_score: Dict[Coord, float] = {start: 0.0}
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        explored_order: List[Coord] = []
        visited: set = set()

        # Priority queue entries: (f_score, tie_breaker_counter, coord)
        # A monotonically increasing counter breaks ties deterministically
        # (avoids comparing Coord tuples when f_scores are equal, and
        # keeps behavior reproducible run-to-run).
        counter = 0
        heap: List[tuple] = [(heuristic_fn(start, goal), counter, start)]

        while heap:
            _, _, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            explored_order.append(current)

            if current == goal:
                break

            for nxt in self.maze.accessible_neighbors(current):
                if nxt in visited:
                    continue
                tentative_g = g_score[current] + 1.0  # unit edge cost
                if nxt not in g_score or tentative_g < g_score[nxt]:
                    g_score[nxt] = tentative_g
                    came_from[nxt] = current
                    f_score = tentative_g + heuristic_fn(nxt, goal)
                    counter += 1
                    heapq.heappush(heap, (f_score, counter, nxt))

        elapsed = time.perf_counter() - t0
        path = _reconstruct_path(came_from, start, goal)

        return SearchResult(
            algorithm="A*",
            path=path,
            path_length=max(len(path) - 1, 0),
            explored_order=explored_order,
            explored_count=len(explored_order),
            computation_time_s=elapsed,
            success=bool(path),
        )

    # ------------------------------------------------------------------ #
    # Convenience: run all four at once
    # ------------------------------------------------------------------ #

    def solve_all(
        self, start: Optional[Coord] = None, goal: Optional[Coord] = None
    ) -> Dict[str, SearchResult]:
        """
        Run all four algorithms (BFS, DFS, Dijkstra, A*) on the same
        start/goal pair and return their results keyed by algorithm name.

        This is the primary entry point used by main.py's "compare
        algorithms" bonus feature.

        Args:
            start: Start coordinate (default: maze.start).
            goal: Goal coordinate (default: maze.finish).

        Returns:
            Dict mapping algorithm name -> SearchResult, in the order
            BFS, DFS, Dijkstra, A*.
        """
        return {
            "BFS": self.solve_bfs(start, goal),
            "DFS": self.solve_dfs(start, goal),
            "Dijkstra": self.solve_dijkstra(start, goal),
            "A*": self.solve_astar(start, goal),
        }
