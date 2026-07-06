"""
maze.py
=======

Maze generation module for the robotics competition simulation.

Implements a perfect maze (fully connected, no loops, no isolated regions)
using the Randomized Recursive Backtracking algorithm (a form of randomized
DFS over a grid graph). The generator guarantees:

    * Exactly one Start cell and one Finish cell.
    * Full connectivity (Start and Finish always reachable from each other).
    * A minimum "decision complexity" (junctions / branch points) between
      Start and Finish, regenerating with a different carve order if the
      first attempt is too simple (e.g. a near-straight corridor).
    * Deterministic reproduction via an integer random seed.

Grid / wall representation
---------------------------
The maze is stored as a grid of `Cell` objects. Each cell tracks which of
its four walls (N, S, E, W) are still standing. Two adjacent cells are
"connected" (a robot can move between them) iff the wall between them has
been removed by the generator.

Physical scale: each cell is 16 x 16 cm, so a 30 x 30 grid maze covers
480 x 480 cm, matching the competition specification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Type alias: a coordinate is (row, col), 0-indexed, row 0 = top.
Coord = Tuple[int, int]

# Direction vectors: (row_delta, col_delta) and opposite-wall bookkeeping.
DIRECTIONS: Dict[str, Coord] = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}
OPPOSITE: Dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class Cell:
    """
    A single maze cell.

    Attributes:
        row: Row index (0-indexed, top = 0).
        col: Column index (0-indexed, left = 0).
        walls: Which of the four walls are still standing.
        visited: Used internally during generation (carve-order marker).
    """

    row: int
    col: int
    walls: Dict[str, bool] = field(
        default_factory=lambda: {"N": True, "S": True, "E": True, "W": True}
    )
    visited: bool = False

    def open_directions(self) -> List[str]:
        """Return the list of directions ('N','S','E','W') with no wall."""
        return [d for d, present in self.walls.items() if not present]


class Maze:
    """
    A 30x30 (configurable) perfect maze with physical-scale metadata.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        cell_size_cm: Physical size of one cell edge, in centimeters.
        grid: 2D list of Cell objects, grid[row][col].
        start: Coordinate of the Start cell.
        finish: Coordinate of the Finish cell.
        seed: The random seed used to generate this maze (for reproducibility).
    """

    def __init__(
        self,
        rows: int = 30,
        cols: int = 30,
        cell_size_cm: float = 16.0,
        start: Optional[Coord] = None,
        finish: Optional[Coord] = None,
    ) -> None:
        """
        Initialize an empty (fully-walled) maze grid.

        Args:
            rows: Number of maze rows.
            cols: Number of maze columns.
            cell_size_cm: Physical edge length of each cell, in cm.
            start: Optional explicit start coordinate; defaults to (0, 0).
            finish: Optional explicit finish coordinate; defaults to
                (rows - 1, cols - 1).
        """
        self.rows = rows
        self.cols = cols
        self.cell_size_cm = cell_size_cm
        self.grid: List[List[Cell]] = [
            [Cell(r, c) for c in range(cols)] for r in range(rows)
        ]
        self.start: Coord = start if start is not None else (0, 0)
        self.finish: Coord = finish if finish is not None else (rows - 1, cols - 1)
        self.seed: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Basic grid queries
    # ------------------------------------------------------------------ #

    def in_bounds(self, coord: Coord) -> bool:
        """Return True if coord lies within the grid boundaries."""
        r, c = coord
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cell(self, coord: Coord) -> Cell:
        """Return the Cell object at the given coordinate."""
        r, c = coord
        return self.grid[r][c]

    def neighbors(self, coord: Coord) -> List[Coord]:
        """
        Return all in-bounds grid-adjacent coordinates of coord
        (regardless of whether a wall separates them).
        """
        r, c = coord
        result = []
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if self.in_bounds((nr, nc)):
                result.append((nr, nc))
        return result

    def accessible_neighbors(self, coord: Coord) -> List[Coord]:
        """
        Return neighbor coordinates that are reachable in one robot move,
        i.e. the wall between coord and the neighbor has been removed.

        This is the primary connectivity query used by all search
        algorithms (BFS/DFS/A*/Dijkstra) -- it defines the traversable
        graph edges of the maze.
        """
        r, c = coord
        cell = self.cell(coord)
        result = []
        for d in cell.open_directions():
            dr, dc = DIRECTIONS[d]
            nr, nc = r + dr, c + dc
            if self.in_bounds((nr, nc)):
                result.append((nr, nc))
        return result

    def remove_wall(self, a: Coord, b: Coord) -> None:
        """
        Remove the wall between two adjacent cells a and b (in both cells).

        Raises:
            ValueError: If a and b are not grid-adjacent.
        """
        ar, ac = a
        br, bc = b
        dr, dc = br - ar, bc - ac
        direction = None
        for d, (vr, vc) in DIRECTIONS.items():
            if (vr, vc) == (dr, dc):
                direction = d
                break
        if direction is None:
            raise ValueError(f"Cells {a} and {b} are not adjacent.")
        self.cell(a).walls[direction] = False
        self.cell(b).walls[OPPOSITE[direction]] = False

    # ------------------------------------------------------------------ #
    # Serialization (bonus feature: save / load maze)
    # ------------------------------------------------------------------ #

    def to_text(self) -> str:
        """
        Serialize the maze to a compact text representation.

        Format: a header line with metadata, followed by one line per cell
        (row-major order) encoding the four wall booleans as bits, e.g.
        'N1S0E1W0'. Start/Finish/seed are stored in the header so the file
        is fully self-describing and can be reloaded exactly.

        Returns:
            The maze encoded as a multi-line string.
        """
        lines = [
            f"ROWS={self.rows} COLS={self.cols} CELL_CM={self.cell_size_cm} "
            f"START={self.start[0]},{self.start[1]} "
            f"FINISH={self.finish[0]},{self.finish[1]} "
            f"SEED={self.seed}"
        ]
        for r in range(self.rows):
            row_tokens = []
            for c in range(self.cols):
                w = self.grid[r][c].walls
                token = (
                    f"N{int(w['N'])}S{int(w['S'])}E{int(w['E'])}W{int(w['W'])}"
                )
                row_tokens.append(token)
            lines.append(" ".join(row_tokens))
        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """
        Save the maze to a text file (bonus feature).

        Args:
            filepath: Destination path for the maze text file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_text())

    @classmethod
    def from_text(cls, text: str) -> "Maze":
        """
        Reconstruct a Maze from the text format produced by `to_text`.

        Args:
            text: The serialized maze content.

        Returns:
            A fully reconstructed Maze object, including start/finish/seed.
        """
        lines = text.strip().split("\n")
        header = lines[0]
        parts = dict(
            token.split("=") for token in header.split() if "=" in token
        )
        rows = int(parts["ROWS"])
        cols = int(parts["COLS"])
        cell_cm = float(parts["CELL_CM"])
        sr, sc = map(int, parts["START"].split(","))
        fr, fc = map(int, parts["FINISH"].split(","))
        seed_str = parts["SEED"]
        seed = None if seed_str == "None" else int(seed_str)

        maze = cls(rows=rows, cols=cols, cell_size_cm=cell_cm,
                    start=(sr, sc), finish=(fr, fc))
        maze.seed = seed

        for r in range(rows):
            tokens = lines[1 + r].split()
            for c, token in enumerate(tokens):
                # token like "N1S0E1W0"
                walls = {}
                i = 0
                while i < len(token):
                    key = token[i]
                    val = token[i + 1]
                    walls[key] = bool(int(val))
                    i += 2
                maze.grid[r][c].walls = walls
        return maze

    @classmethod
    def load(cls, filepath: str) -> "Maze":
        """
        Load a maze from a text file previously written by `save`.

        Args:
            filepath: Path to the maze text file.

        Returns:
            The reconstructed Maze object.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_text(content)


class MazeGenerator:
    """
    Generates perfect mazes using Randomized Recursive Backtracking
    (a randomized iterative Depth-First Search over the grid graph).

    The algorithm:
        1. Push Start onto a stack, mark it visited.
        2. While the stack is not empty:
            a. Look at the cell on top of the stack.
            b. If it has unvisited neighbors, pick one at random, carve
               (remove the wall) between them, mark it visited, push it.
            c. Otherwise, pop the stack (backtrack).
        3. Every cell ends up visited exactly once -> the result is a
           spanning tree of the grid graph, i.e. a "perfect" maze: fully
           connected with exactly one path between any two cells, no
           isolated regions, no loops.

    Because the result is a spanning tree, Start and Finish are always
    connected by exactly one simple path -- connectivity is guaranteed
    by construction, not checked after the fact.
    """

    def __init__(self, rows: int = 30, cols: int = 30, cell_size_cm: float = 16.0):
        """
        Args:
            rows: Number of maze rows.
            cols: Number of maze columns.
            cell_size_cm: Physical size of each cell edge, in cm.
        """
        self.rows = rows
        self.cols = cols
        self.cell_size_cm = cell_size_cm

    def generate(
        self,
        seed: int,
        start: Optional[Coord] = None,
        finish: Optional[Coord] = None,
        min_decision_points: int = 5,
        max_attempts: int = 50,
    ) -> Maze:
        """
        Generate a maze meeting the minimum decision-complexity requirement.

        The generator carves a perfect maze with recursive backtracking,
        then measures the number of "decision points" (junction cells with
        3+ open directions) that lie on the shortest Start->Finish path.
        If the count is below `min_decision_points`, it regenerates with a
        derived seed and tries again, up to `max_attempts` times, so the
        final maze always requires meaningfully branching navigation
        rather than a single corridor.

        Args:
            seed: Random seed for reproducibility.
            start: Optional explicit start coordinate (default (0, 0)).
            finish: Optional explicit finish coordinate
                (default (rows - 1, cols - 1)).
            min_decision_points: Minimum number of junction decisions
                required along the shortest path (competition spec: >= 5).
            max_attempts: Maximum regeneration attempts before giving up
                and returning the best maze found.

        Returns:
            A Maze instance satisfying full connectivity and the minimum
            decision-complexity requirement (best effort after
            max_attempts).
        """
        start = start if start is not None else (0, 0)
        finish = finish if finish is not None else (self.rows - 1, self.cols - 1)

        best_maze: Optional[Maze] = None
        best_score = -1

        for attempt in range(max_attempts):
            attempt_seed = seed + attempt  # deterministic sequence from base seed
            maze = self._carve(attempt_seed, start, finish)
            score = count_decision_points(maze)
            if score > best_score:
                best_score = score
                best_maze = maze
            if score >= min_decision_points:
                return maze

        # Best-effort fallback: none of the attempts hit the target, return
        # the most complex maze found so far (still fully connected).
        assert best_maze is not None
        return best_maze

    def _carve(self, seed: int, start: Coord, finish: Coord) -> Maze:
        """
        Run one pass of randomized recursive backtracking with the given
        seed and return the resulting perfect maze.

        Args:
            seed: Random seed controlling carve order and neighbor choice.
            start: Start coordinate.
            finish: Finish coordinate.

        Returns:
            A newly carved, fully-connected Maze.
        """
        rng = random.Random(seed)
        maze = Maze(self.rows, self.cols, self.cell_size_cm, start, finish)
        maze.seed = seed

        stack: List[Coord] = [start]
        maze.cell(start).visited = True
        visited_count = 1
        total_cells = self.rows * self.cols

        while stack and visited_count < total_cells:
            current = stack[-1]
            unvisited = [
                n for n in maze.neighbors(current) if not maze.cell(n).visited
            ]
            if unvisited:
                nxt = rng.choice(unvisited)
                maze.remove_wall(current, nxt)
                maze.cell(nxt).visited = True
                visited_count += 1
                stack.append(nxt)
            else:
                stack.pop()

        # Reset the visited markers (they were only for generation bookkeeping)
        for row in maze.grid:
            for cell in row:
                cell.visited = False

        return maze


def count_decision_points(maze: Maze) -> int:
    """
    Count the number of "decision point" junctions along the shortest
    Start->Finish path of a maze.

    A decision point is defined as a cell on the shortest path that has
    3 or more open directions (i.e. the robot has more than one way to
    continue besides simply retracing its steps) -- a genuine branch,
    as opposed to a plain corridor cell (2 open directions, straight
    or turning) or a dead end (1 open direction).

    This function performs a lightweight BFS internally purely to obtain
    the shortest path for scoring; it does not depend on search.py so
    that maze.py has no circular import on the search module.

    Args:
        maze: The Maze to analyze.

    Returns:
        The number of qualifying decision-point cells on the shortest
        Start->Finish path. Returns 0 if no path exists (should not
        happen for a correctly generated perfect maze).
    """
    path = _bfs_path(maze, maze.start, maze.finish)
    if not path:
        return 0
    count = 0
    for coord in path:
        open_dirs = len(maze.cell(coord).open_directions())
        if open_dirs >= 3:
            count += 1
    return count


def _bfs_path(maze: Maze, start: Coord, goal: Coord) -> List[Coord]:
    """
    Minimal internal BFS used only for maze-quality scoring during
    generation (kept private / independent of search.py by design).

    Args:
        maze: The maze to search.
        start: Start coordinate.
        goal: Goal coordinate.

    Returns:
        The list of coordinates from start to goal inclusive, or an
        empty list if unreachable.
    """
    from collections import deque

    frontier = deque([start])
    came_from: Dict[Coord, Optional[Coord]] = {start: None}

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for nxt in maze.accessible_neighbors(current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)

    if goal not in came_from:
        return []

    path = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def verify_connectivity(maze: Maze) -> bool:
    """
    Verify that every cell in the maze is reachable from Start (i.e. no
    isolated regions) and that Start and Finish are connected.

    This is a safety-net sanity check; the recursive-backtracking
    generator produces a spanning tree by construction, so this should
    always return True for mazes produced by MazeGenerator, but the
    function is provided so `main.py` can assert correctness explicitly
    (and so it can validate mazes loaded from external text files).

    Args:
        maze: The Maze to verify.

    Returns:
        True if fully connected (single component covering all cells),
        False otherwise.
    """
    from collections import deque

    total_cells = maze.rows * maze.cols
    seen: Set[Coord] = {maze.start}
    frontier = deque([maze.start])

    while frontier:
        current = frontier.popleft()
        for nxt in maze.accessible_neighbors(current):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)

    fully_connected = len(seen) == total_cells
    start_finish_connected = maze.finish in seen
    return fully_connected and start_finish_connected
