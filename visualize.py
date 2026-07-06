"""
visualize.py
============

Visualization module for the maze robot simulation, built on matplotlib.

Renders:
    1. The maze walls/corridors.
    2. Start cell.
    3. Finish cell.
    4. Cheese (at Finish).
    5. The robot's planned path.
    6. Visited/explored cells (search "footprint").

Also provides:
    * A side-by-side algorithm comparison figure (bonus feature).
    * A simple frame-by-frame animation of the robot walking its path
      (bonus feature), using matplotlib.animation.FuncAnimation.

Color scheme (deliberately chosen, not matplotlib defaults):
    * Background / unvisited corridor : warm off-white  (#FBF7F0)
    * Walls                            : charcoal        (#2B2B2B)
    * Start                            : deep teal        (#0F766E)
    * Finish                           : amber            (#D97706)
    * Cheese marker                    : golden yellow    (#FACC15)
    * Explored / visited cells         : dusty lavender   (#C7B8EA), translucent
    * Final path                       : deep indigo      (#4C1D95)
    * Robot marker                     : crimson          (#B91C1C)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from maze import Coord, Maze
from search import SearchResult

# ---------------------------------------------------------------------- #
# Palette (named, deliberate -- see module docstring)
# ---------------------------------------------------------------------- #
COLOR_BACKGROUND = "#FBF7F0"
COLOR_WALL = "#2B2B2B"
COLOR_START = "#0F766E"
COLOR_FINISH = "#D97706"
COLOR_CHEESE = "#FACC15"
COLOR_VISITED = "#C7B8EA"
COLOR_PATH = "#4C1D95"
COLOR_ROBOT = "#B91C1C"


def _draw_maze_walls(ax: plt.Axes, maze: Maze) -> None:
    """
    Draw the maze's walls onto the given matplotlib Axes as line segments.

    Cell (row, col) occupies the unit square
    [col, col+1] x [-(row+1), -row] in plot coordinates (row increases
    downward on screen, matching how a grid is normally read top-to-bottom,
    but plotted with row 0 at the top of the figure).

    Args:
        ax: The matplotlib Axes to draw on.
        maze: The Maze whose walls should be rendered.
    """
    segments = []
    for r in range(maze.rows):
        for c in range(maze.cols):
            walls = maze.grid[r][c].walls
            x0, x1 = c, c + 1
            y0, y1 = -(r + 1), -r  # y0 = bottom edge, y1 = top edge of this cell

            if walls["N"]:
                segments.append([(x0, y1), (x1, y1)])
            if walls["S"]:
                segments.append([(x0, y0), (x1, y0)])
            if walls["W"]:
                segments.append([(x0, y0), (x0, y1)])
            if walls["E"]:
                segments.append([(x1, y0), (x1, y1)])

    wall_collection = LineCollection(segments, colors=COLOR_WALL, linewidths=1.6)
    ax.add_collection(wall_collection)


def _cell_center(coord: Coord) -> Coord:
    """
    Return the (x, y) plot-coordinate center of a given maze cell,
    matching the coordinate convention used in `_draw_maze_walls`.

    Args:
        coord: (row, col) maze coordinate.

    Returns:
        (x, y) tuple of plot coordinates for the cell's center.
    """
    r, c = coord
    return (c + 0.5, -(r + 0.5))


def _highlight_cells(
    ax: plt.Axes, cells: Sequence[Coord], color: str, alpha: float, zorder: int
) -> None:
    """
    Fill a set of maze cells with a translucent colored rectangle patch.

    Used for rendering the "visited cells" search footprint.

    Args:
        ax: The matplotlib Axes to draw on.
        cells: Iterable of (row, col) coordinates to highlight.
        color: Fill color (hex string).
        alpha: Fill transparency (0 = invisible, 1 = opaque).
        zorder: Matplotlib draw order (higher draws on top).
    """
    for r, c in cells:
        rect = mpatches.Rectangle(
            (c, -(r + 1)), 1, 1, facecolor=color, edgecolor="none",
            alpha=alpha, zorder=zorder,
        )
        ax.add_patch(rect)


def _draw_path_line(ax: plt.Axes, path: Sequence[Coord], color: str, zorder: int) -> None:
    """
    Draw the robot's path as a connected line through cell centers.

    Args:
        ax: The matplotlib Axes to draw on.
        path: Ordered sequence of (row, col) coordinates.
        color: Line color (hex string).
        zorder: Matplotlib draw order (higher draws on top).
    """
    if len(path) < 2:
        return
    xs = [_cell_center(p)[0] for p in path]
    ys = [_cell_center(p)[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=2.8, solid_capstyle="round", zorder=zorder)


def render_maze(
    maze: Maze,
    path: Optional[Sequence[Coord]] = None,
    visited: Optional[Sequence[Coord]] = None,
    title: str = "Maze Simulation",
    ax: Optional[plt.Axes] = None,
    show_cheese: bool = True,
) -> plt.Axes:
    """
    Render a single maze figure with all required visual elements:
    walls, Start, Finish, cheese, robot path, and visited cells.

    Args:
        maze: The Maze to render.
        path: Optional ordered path to draw (e.g. the chosen algorithm's
            solution).
        visited: Optional collection of explored/visited cells to
            highlight as the search "footprint".
        title: Figure/axes title.
        ax: Optional existing Axes to draw into (for composing multi-
            panel figures). If None, a new standalone figure is created.
        show_cheese: Whether to draw the cheese marker at Finish.

    Returns:
        The matplotlib Axes the maze was drawn on (useful for further
        customization or for saving the parent figure).
    """
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor(COLOR_BACKGROUND)

    ax.set_facecolor(COLOR_BACKGROUND)
    ax.set_xlim(0, maze.cols)
    ax.set_ylim(-maze.rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 6. Visited cells (drawn first so path/markers layer on top)
    if visited:
        _highlight_cells(ax, visited, COLOR_VISITED, alpha=0.55, zorder=2)

    # 1. Maze walls
    _draw_maze_walls(ax, maze)

    # 5. Robot path
    if path:
        _draw_path_line(ax, path, COLOR_PATH, zorder=4)

    # 2. Start marker
    sx, sy = _cell_center(maze.start)
    ax.scatter([sx], [sy], s=260, marker="s", color=COLOR_START, zorder=5,
               edgecolors="white", linewidths=1.2)

    # 3. Finish marker
    fx, fy = _cell_center(maze.finish)
    ax.scatter([fx], [fy], s=260, marker="D", color=COLOR_FINISH, zorder=5,
               edgecolors="white", linewidths=1.2)

    # 4. Cheese marker (slightly offset star on top of Finish)
    if show_cheese:
        ax.scatter([fx], [fy], s=110, marker="*", color=COLOR_CHEESE, zorder=6,
                   edgecolors=COLOR_WALL, linewidths=0.6)

    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_WALL, pad=10)

    # Legend (only build once, on the owning figure, to avoid clutter in
    # multi-panel comparisons)
    if owns_figure:
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", label="Start",
                   markerfacecolor=COLOR_START, markersize=12),
            Line2D([0], [0], marker="D", color="w", label="Finish",
                   markerfacecolor=COLOR_FINISH, markersize=11),
            Line2D([0], [0], marker="*", color="w", label="Cheese",
                   markerfacecolor=COLOR_CHEESE, markersize=15),
            Line2D([0], [0], color=COLOR_PATH, lw=3, label="Robot path"),
            mpatches.Patch(facecolor=COLOR_VISITED, alpha=0.55, label="Visited cells"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper center",
            bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=9,
        )

    return ax


def render_comparison(
    maze: Maze, results: Dict[str, SearchResult], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Render a 2x2 comparison figure showing all four algorithms' paths and
    explored footprints side by side (bonus feature: compare search
    algorithms visually).

    Args:
        maze: The Maze that was solved.
        results: Dict of algorithm name -> SearchResult (as returned by
            MazeSolver.solve_all).
        save_path: Optional filepath to save the figure to (e.g. PNG).

    Returns:
        The matplotlib Figure containing the 2x2 comparison grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    axes_flat = axes.flatten()

    order = ["BFS", "DFS", "Dijkstra", "A*"]
    for ax, name in zip(axes_flat, order):
        result = results[name]
        subtitle = (
            f"{name}  |  len={result.path_length}  "
            f"explored={result.explored_count}  "
            f"t={result.computation_time_s * 1000:.2f} ms"
        )
        render_maze(
            maze, path=result.path, visited=result.explored_order,
            title=subtitle, ax=ax,
        )

    fig.suptitle(
        "Search Algorithm Comparison", fontsize=17, fontweight="bold",
        color=COLOR_WALL, y=0.98,
    )

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label="Start",
               markerfacecolor=COLOR_START, markersize=12),
        Line2D([0], [0], marker="D", color="w", label="Finish",
               markerfacecolor=COLOR_FINISH, markersize=11),
        Line2D([0], [0], marker="*", color="w", label="Cheese",
               markerfacecolor=COLOR_CHEESE, markersize=15),
        Line2D([0], [0], color=COLOR_PATH, lw=3, label="Robot path"),
        mpatches.Patch(facecolor=COLOR_VISITED, alpha=0.55, label="Visited cells"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    return fig


def animate_robot(
    maze: Maze,
    path: Sequence[Coord],
    visited: Optional[Sequence[Coord]] = None,
    title: str = "Robot Navigation",
    interval_ms: int = 120,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Animate the robot walking along `path`, one cell per frame
    (bonus feature).

    Args:
        maze: The Maze being navigated.
        path: Ordered sequence of coordinates the robot walks through.
        visited: Optional explored-cell footprint to display as static
            background context throughout the animation.
        title: Animation title.
        interval_ms: Delay between animation frames, in milliseconds.
        save_path: Optional filepath to save the animation (e.g. as a
            .gif or .mp4, depending on available matplotlib writers).

    Returns:
        The matplotlib FuncAnimation object. Assign it to a variable in
        interactive contexts to prevent garbage collection from stopping
        the animation prematurely.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(COLOR_BACKGROUND)

    render_maze(maze, path=None, visited=visited, title=title, ax=ax)

    robot_marker = ax.scatter(
        [], [], s=300, marker="o", color=COLOR_ROBOT, zorder=7,
        edgecolors="white", linewidths=1.5,
    )
    trail_line, = ax.plot([], [], color=COLOR_PATH, linewidth=2.8,
                          solid_capstyle="round", zorder=4)

    def _init():
        robot_marker.set_offsets([[0, 0]])
        robot_marker.set_visible(False)
        trail_line.set_data([], [])
        return robot_marker, trail_line

    def _update(frame_idx: int):
        current_path = path[: frame_idx + 1]
        xs = [_cell_center(p)[0] for p in current_path]
        ys = [_cell_center(p)[1] for p in current_path]
        trail_line.set_data(xs, ys)

        cx, cy = _cell_center(path[frame_idx])
        robot_marker.set_offsets([[cx, cy]])
        robot_marker.set_visible(True)
        return robot_marker, trail_line

    anim = FuncAnimation(
        fig, _update, frames=len(path), init_func=_init,
        interval=interval_ms, blit=True, repeat=False,
    )

    if save_path:
        try:
            anim.save(save_path, dpi=120)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(
                f"[visualize] Could not save animation to {save_path} "
                f"(missing writer backend?): {exc}"
            )

    return anim


def render_distance_heatmap(
    maze: Maze, distance_grid: List[List[int]], title: str = "Cheese Distance Map"
) -> plt.Figure:
    """
    Render the cheese distance map as a heatmap overlaid with maze walls
    (bonus visualization: shows the heuristic/sensing data directly).

    Args:
        maze: The Maze whose distance map is being visualized.
        distance_grid: 2D list (rows x cols) of shortest-path distances
            to Finish, as produced by CheeseSensor.as_grid().
        title: Figure title.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=(9.5, 9))
    fig.patch.set_facecolor(COLOR_BACKGROUND)

    im = ax.imshow(
        distance_grid,
        cmap="YlOrRd_r",
        extent=(0, maze.cols, -maze.rows, 0),
        interpolation="nearest",
        zorder=1,
    )
    _draw_maze_walls(ax, maze)

    fx, fy = _cell_center(maze.finish)
    ax.scatter([fx], [fy], s=200, marker="*", color=COLOR_CHEESE, zorder=6,
               edgecolors=COLOR_WALL, linewidths=0.8)
    sx, sy = _cell_center(maze.start)
    ax.scatter([sx], [sy], s=220, marker="s", color=COLOR_START, zorder=5,
               edgecolors="white", linewidths=1.2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_WALL, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Distance to cheese (grid steps)", fontsize=10)

    return fig
