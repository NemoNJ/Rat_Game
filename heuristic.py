"""
heuristic.py
============

Cheese-sensing / heuristic distance module.

The competition mouse can "smell" the cheese at the Finish cell using the
true shortest-path (graph) distance from every reachable cell to Finish.
This module computes that full distance map with a single multi-source-
style BFS run backwards from Finish (equivalent to a forwards BFS since
maze connectivity is symmetric/undirected), giving an O(V) exact distance
to the goal for every cell in one pass.

The resulting distance map is used in two ways elsewhere in the program:
    1. As ground truth for computing each algorithm's returned path length.
    2. As the admissible heuristic function h(n) for A* (Manhattan distance
       is used online during search since the true map would be "cheating"
       information for a real robot; the exact map is provided here mainly
       as the "cheese smell" sensing feature requested by the spec, and to
       let main.py display/verify how far every cell is from the goal).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

from maze import Coord, Maze


class CheeseSensor:
    """
    Computes and stores the shortest-path distance from every reachable
    cell in the maze to the Finish cell (where the cheese is located).

    Attributes:
        maze: The Maze instance being sensed.
        distance_map: Dict mapping coord -> shortest number of moves to
            Finish. Unreachable cells (should not occur in a connected
            maze) are simply absent from the map.
    """

    def __init__(self, maze: Maze) -> None:
        """
        Args:
            maze: The maze containing the cheese at maze.finish.
        """
        self.maze = maze
        self.distance_map: Dict[Coord, int] = {}

    def compute_distance_map(self) -> Dict[Coord, int]:
        """
        Compute the shortest-path distance from every reachable cell to
        the Finish (cheese) cell using a single BFS seeded at Finish.

        Because maze connectivity is undirected (a removed wall connects
        both cells symmetrically), BFS from Finish yields the same
        distances as BFS from any cell TO Finish -- this lets us compute
        the distance for *every* cell in one O(V) pass instead of running
        a separate search per cell.

        Returns:
            Dict mapping each reachable coordinate to its shortest-path
            distance (in grid steps) to the cheese at Finish.
        """
        goal = self.maze.finish
        distances: Dict[Coord, int] = {goal: 0}
        frontier = deque([goal])

        while frontier:
            current = frontier.popleft()
            current_dist = distances[current]
            for neighbor in self.maze.accessible_neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    frontier.append(neighbor)

        self.distance_map = distances
        return distances

    def smell_distance(self, coord: Coord) -> Optional[int]:
        """
        Return how far (in grid steps) the mouse can smell the cheese
        from the given cell.

        Args:
            coord: The cell to query.

        Returns:
            The shortest-path distance in grid steps to Finish, or None
            if the cell cannot reach Finish (should not happen in a
            correctly generated, fully-connected maze) or if
            `compute_distance_map` has not yet been called.
        """
        if not self.distance_map:
            self.compute_distance_map()
        return self.distance_map.get(coord)

    def as_grid(self) -> list:
        """
        Return the distance map as a 2D list matching the maze grid
        dimensions, convenient for visualization (e.g. as a heatmap) or
        printing. Unreachable cells are represented as -1.

        Returns:
            A rows x cols nested list of integers (distance in steps,
            or -1 if unreachable).
        """
        if not self.distance_map:
            self.compute_distance_map()
        grid = [[-1 for _ in range(self.maze.cols)] for _ in range(self.maze.rows)]
        for (r, c), dist in self.distance_map.items():
            grid[r][c] = dist
        return grid


def manhattan_distance(a: Coord, b: Coord) -> int:
    """
    Compute the Manhattan (L1) distance between two grid coordinates.

    This is the admissible, consistent heuristic used online by the A*
    search algorithm (see search.py). It is admissible because the robot
    can only move one cell at a time along grid axes (no diagonals), so
    Manhattan distance never overestimates the true remaining cost --
    walls can only make the true path longer than this, never shorter.

    Args:
        a: First coordinate (row, col).
        b: Second coordinate (row, col).

    Returns:
        The Manhattan distance |a.row - b.row| + |a.col - b.col|.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
