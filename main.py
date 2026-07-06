"""
main.py
=======

Entry point for the robotics competition maze simulation.

Orchestrates the full pipeline:
    1. Generate a 30x30 maze (16x16 cm cells -> 480x480 cm arena) with a
       reproducible random seed, guaranteeing full connectivity and a
       minimum of 5 navigation decision points between Start and Finish.
    2. Compute the cheese-sensing distance map (shortest-path distance
       from every cell to Finish).
    3. Solve the maze with BFS, DFS, Dijkstra, and A*.
    4. Verify the 3-minute per-move thinking-time constraint is satisfied
       (trivially, given real computation times).
    5. Print performance statistics comparing all four algorithms.
    6. Visualize the maze, chosen path, and visited cells.
    7. (Bonus) Save/load the maze to/from a text file, render an
       algorithm-comparison figure, and animate the robot's traversal.

Run directly:
    python main.py
    python main.py --seed 42 --no-show
    python main.py --seed 7 --algorithm astar --animate
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import matplotlib.pyplot as plt

from heuristic import CheeseSensor
from maze import Maze, MazeGenerator, count_decision_points, verify_connectivity
from search import MazeSolver, SearchResult
from visualize import animate_robot, render_comparison, render_distance_heatmap, render_maze

# Competition constraint: the robot "dies" if a single decision takes
# longer than this many seconds to compute.
MAX_THINKING_TIME_S = 180.0

# Competition constraint: the maze must require at least this many
# genuine navigation decision points along the shortest path.
MIN_DECISION_POINTS = 5

ALGORITHM_DISPATCH = {
    "bfs": "solve_bfs",
    "dfs": "solve_dfs",
    "dijkstra": "solve_dijkstra",
    "astar": "solve_astar",
}


def build_maze(seed: int, rows: int = 30, cols: int = 30, cell_cm: float = 16.0) -> Maze:
    """
    Generate a competition-ready maze and assert all hard requirements.

    Args:
        seed: Random seed for reproducible generation.
        rows: Number of maze rows (default 30, per spec).
        cols: Number of maze columns (default 30, per spec).
        cell_cm: Physical cell edge length in cm (default 16, per spec).

    Returns:
        A validated Maze instance.

    Raises:
        AssertionError: If the generated maze fails connectivity or
            decision-complexity requirements (should not occur in
            practice; this is a defensive sanity check).
    """
    generator = MazeGenerator(rows=rows, cols=cols, cell_size_cm=cell_cm)
    maze = generator.generate(seed=seed, min_decision_points=MIN_DECISION_POINTS)

    assert verify_connectivity(maze), (
        "Generated maze failed connectivity check -- Start/Finish must "
        "always be connected with no isolated regions."
    )
    decision_points = count_decision_points(maze)
    if decision_points < MIN_DECISION_POINTS:
        print(
            f"[warning] Best maze found has {decision_points} decision "
            f"points (target was {MIN_DECISION_POINTS}). Proceeding anyway "
            f"with the most complex maze found."
        )

    return maze


def print_maze_info(maze: Maze) -> None:
    """
    Print a human-readable summary of the maze's physical and structural
    properties.

    Args:
        maze: The Maze to describe.
    """
    width_cm = maze.cols * maze.cell_size_cm
    height_cm = maze.rows * maze.cell_size_cm
    decision_points = count_decision_points(maze)

    print("=" * 70)
    print("MAZE CONFIGURATION")
    print("=" * 70)
    print(f"Grid size            : {maze.rows} x {maze.cols} cells")
    print(f"Cell size             : {maze.cell_size_cm:.1f} x {maze.cell_size_cm:.1f} cm")
    print(f"Physical arena size   : {width_cm:.1f} x {height_cm:.1f} cm")
    print(f"Start                 : {maze.start}")
    print(f"Finish (cheese)       : {maze.finish}")
    print(f"Random seed           : {maze.seed}")
    print(f"Fully connected       : {verify_connectivity(maze)}")
    print(f"Decision points on    : {decision_points} "
          f"(minimum required: {MIN_DECISION_POINTS})")
    print("  shortest path")
    print("=" * 70)


def print_performance_statistics(results: Dict[str, SearchResult]) -> None:
    """
    Print a formatted performance comparison table across all algorithms,
    including an explicit 3-minute thinking-time compliance check.

    Args:
        results: Dict of algorithm name -> SearchResult.
    """
    print()
    print("=" * 70)
    print("SEARCH ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 70)
    header = f"{'Algorithm':<10} {'Success':<9} {'PathLen':<9} {'Explored':<10} {'Time(ms)':<12} {'ThinkOK':<8}"
    print(header)
    print("-" * 70)

    for name, result in results.items():
        think_ok = "YES" if result.computation_time_s <= MAX_THINKING_TIME_S else "NO"
        print(
            f"{name:<10} {str(result.success):<9} {result.path_length:<9} "
            f"{result.explored_count:<10} {result.computation_time_s * 1000:<12.4f} "
            f"{think_ok:<8}"
        )

    print("-" * 70)

    optimal_len = min(
        (r.path_length for r in results.values() if r.success), default=None
    )
    if optimal_len is not None:
        print(f"Optimal (shortest) path length : {optimal_len} moves")
        for name, result in results.items():
            if result.success:
                tag = "OPTIMAL" if result.path_length == optimal_len else "SUBOPTIMAL"
                print(f"  {name:<10}: {result.path_length} moves  [{tag}]")

    fastest = min(results.values(), key=lambda r: r.computation_time_s)
    most_efficient = min(
        (r for r in results.values() if r.success), key=lambda r: r.explored_count,
        default=None,
    )
    print()
    print(f"Fastest computation      : {fastest.algorithm} "
          f"({fastest.computation_time_s * 1000:.4f} ms)")
    if most_efficient is not None:
        print(f"Fewest cells explored     : {most_efficient.algorithm} "
              f"({most_efficient.explored_count} cells)")
    print("=" * 70)


def verify_thinking_time(results: Dict[str, SearchResult]) -> bool:
    """
    Confirm every algorithm's computation time is within the competition's
    3-minute (180 second) per-move thinking-time limit.

    Args:
        results: Dict of algorithm name -> SearchResult.

    Returns:
        True if all algorithms complied with the time limit, False
        otherwise (the robot would be considered "dead").
    """
    all_ok = True
    for name, result in results.items():
        if result.computation_time_s > MAX_THINKING_TIME_S:
            print(
                f"[CRITICAL] {name} exceeded the {MAX_THINKING_TIME_S:.0f}s "
                f"thinking-time limit ({result.computation_time_s:.2f}s) -- "
                f"robot would be considered dead."
            )
            all_ok = False
    return all_ok


def print_cheese_sensing_sample(maze: Maze, sensor: CheeseSensor, sample_size: int = 5) -> None:
    """
    Print a small sample of the cheese distance map to demonstrate the
    sensing feature, including Start's smell-distance to the goal.

    Args:
        maze: The Maze being sensed.
        sensor: A CheeseSensor with `compute_distance_map` already run
            (or about to be run on first access).
        sample_size: How many arbitrary cells to sample and print,
            beyond Start and Finish (which are always shown).
    """
    print()
    print("=" * 70)
    print("CHEESE SENSING (shortest-path distance to Finish)")
    print("=" * 70)
    start_dist = sensor.smell_distance(maze.start)
    finish_dist = sensor.smell_distance(maze.finish)
    print(f"Distance from Start  {maze.start} to cheese : {start_dist} moves")
    print(f"Distance from Finish {maze.finish} to cheese : {finish_dist} moves "
          f"(should be 0)")

    sampled = 0
    for coord, dist in sensor.distance_map.items():
        if coord in (maze.start, maze.finish):
            continue
        print(f"  Sample cell {coord} -> {dist} moves to cheese")
        sampled += 1
        if sampled >= sample_size:
            break
    print("=" * 70)


def run_simulation(
    seed: int = 42,
    algorithm: str = "astar",
    show_plots: bool = True,
    save_maze_path: str = "maze_output.txt",
    load_maze_path: str = "",
    compare_all: bool = True,
    animate: bool = False,
    show_heatmap: bool = False,
) -> None:
    """
    Run the full end-to-end maze simulation pipeline.

    Args:
        seed: Random seed for reproducible maze generation.
        algorithm: Which single algorithm's path to feature in the main
            visualization ('bfs', 'dfs', 'dijkstra', or 'astar').
        show_plots: Whether to call plt.show() to display figures
            interactively (set False for headless / automated runs).
        save_maze_path: Filepath to save the generated maze to (bonus
            feature); pass "" to skip saving.
        load_maze_path: If non-empty, load the maze from this filepath
            instead of generating a new one (bonus feature).
        compare_all: Whether to also run and visualize all four
            algorithms side by side (bonus feature).
        animate: Whether to render a frame-by-frame animation of the
            robot walking the chosen algorithm's path (bonus feature).
        show_heatmap: Whether to render the cheese distance map as a
            heatmap (bonus visualization).
    """
    # ------------------------------------------------------------------ #
    # 1. Maze generation (or loading)
    # ------------------------------------------------------------------ #
    if load_maze_path:
        print(f"Loading maze from '{load_maze_path}' ...")
        maze = Maze.load(load_maze_path)
        assert verify_connectivity(maze), "Loaded maze failed connectivity check."
    else:
        print(f"Generating a new 30x30 maze with seed={seed} ...")
        maze = build_maze(seed=seed)

    print_maze_info(maze)

    if save_maze_path:
        maze.save(save_maze_path)
        print(f"\nMaze saved to '{save_maze_path}'.")

    # ------------------------------------------------------------------ #
    # 2. Cheese sensing (distance-to-goal map)
    # ------------------------------------------------------------------ #
    sensor = CheeseSensor(maze)
    sensor.compute_distance_map()
    print_cheese_sensing_sample(maze, sensor)

    # ------------------------------------------------------------------ #
    # 3. Path planning
    # ------------------------------------------------------------------ #
    solver = MazeSolver(maze)
    all_results = solver.solve_all()

    for result in all_results.values():
        print(result.summary())

    # ------------------------------------------------------------------ #
    # 4. Thinking-time constraint verification
    # ------------------------------------------------------------------ #
    thinking_ok = verify_thinking_time(all_results)
    print(f"\n3-minute thinking-time constraint satisfied: {thinking_ok}")

    # ------------------------------------------------------------------ #
    # 5. Performance statistics
    # ------------------------------------------------------------------ #
    print_performance_statistics(all_results)

    # ------------------------------------------------------------------ #
    # 6. Visualization
    # ------------------------------------------------------------------ #
    algo_key = algorithm.lower()
    if algo_key not in ALGORITHM_DISPATCH:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: "
            f"{list(ALGORITHM_DISPATCH.keys())}"
        )
    algo_name_map = {"bfs": "BFS", "dfs": "DFS", "dijkstra": "Dijkstra", "astar": "A*"}
    featured_result = all_results[algo_name_map[algo_key]]

    render_maze(
        maze,
        path=featured_result.path,
        visited=featured_result.explored_order,
        title=f"Maze Simulation -- Featured Algorithm: {featured_result.algorithm}",
    )

    if compare_all:
        render_comparison(maze, all_results)

    if show_heatmap:
        render_distance_heatmap(maze, sensor.as_grid())

    if animate and featured_result.success:
        print(
            f"\nAnimating robot traversal along the {featured_result.algorithm} "
            f"path ({featured_result.path_length} moves) ..."
        )
        animate_robot(
            maze,
            featured_result.path,
            visited=featured_result.explored_order,
            title=f"Robot Navigation -- {featured_result.algorithm}",
        )

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse command-line arguments for the simulation.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="30x30 maze robotics competition simulation "
        "(BFS / DFS / Dijkstra / A* path planning with cheese sensing)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible maze generation (default: 42).",
    )
    parser.add_argument(
        "--algorithm", type=str, default="astar",
        choices=list(ALGORITHM_DISPATCH.keys()),
        help="Which algorithm's path to feature in the primary "
        "visualization (default: astar).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display matplotlib windows (useful for headless runs).",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Skip the 2x2 all-algorithms comparison figure.",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Animate the robot walking the featured algorithm's path.",
    )
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Render the cheese distance map as a heatmap.",
    )
    parser.add_argument(
        "--save-maze", type=str, default="maze_output.txt",
        help="Filepath to save the generated maze as text "
        "(default: maze_output.txt; pass '' to skip).",
    )
    parser.add_argument(
        "--load-maze", type=str, default="",
        help="Filepath to load a previously saved maze from, instead of "
        "generating a new one.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_simulation(
        seed=args.seed,
        algorithm=args.algorithm,
        show_plots=not args.no_show,
        save_maze_path=args.save_maze,
        load_maze_path=args.load_maze,
        compare_all=not args.no_compare,
        animate=args.animate,
        show_heatmap=args.heatmap,
    )


if __name__ == "__main__":
    main()
