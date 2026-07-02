"""Main CLI entrypoint for training, demo, and analysis.

Run examples:
    uv run python -m royal_game_of_ur --demo
    uv run python -m royal_game_of_ur --output-dir artifacts
    uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_3
"""

from __future__ import annotations
 
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
 
from .environment import RoyalGameOfUrEnv
from .training import plot_training_result, train_sarsa_lambda
from .analysis import (
    plot_all_initial_q,
    plot_win_rate_baseline,
    plot_lambda_sweep,
    plot_alpha_sweep,
    plot_episode_sweep,
    plot_dice_comparison,
)
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Train SARSA(lambda) on the Royal Game of Ur environment.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.8, help="Eligibility trace decay.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for generated plots.")
    parser.add_argument("--demo", action="store_true", help="Run a short random-play verification demo instead of training.")
    parser.add_argument("--analyse", action="store_true", help="Run Exercise 4 convergence analysis (λ and α sweeps).")  # ← new
    parser.add_argument("--skip-lambda-sweep", action="store_true", help="With --analyse: skip the λ sweep.")           # ← new
    parser.add_argument("--skip-alpha-sweep",  action="store_true", help="With --analyse: skip the α sweep.")           # ← new
    parser.add_argument("--skip-episode-sweep", action="store_true", help="With --analyse: skip the episode-count sweep.")
    parser.add_argument(
        "--parallel-sweeps",
        action="store_true",
        help="With --analyse: run λ and α sweeps in parallel when both are enabled.",
    )
    args = parser.parse_args()
 
    # ------------------------------------------------------------------ demo
    if args.demo:
        env = RoyalGameOfUrEnv()
        observation, info = env.reset(seed=args.seed)
 
        print("Royal Game of Ur Gymnasium environment ready")
        print(f"Initial observation: {observation}")
        print(f"Initial info: {info}")
 
        terminated = False
        truncated = False
        step_count = 0
 
        while not terminated and not truncated and step_count < 25:
            action = env.sample_action()
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            print(
                f"step={step_count} action={action} reward={reward} "
                f"terminated={terminated} truncated={truncated} info={info}"
            )
 
        env.close()
        return 0
 
    # ----------------------------------------------------------------- analyse
    if args.analyse:                                                         
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        ep = args.episodes
 
        print(f"=== Baseline plots ({ep:,} episodes) ===")
        baseline_env = RoyalGameOfUrEnv()
        baseline_result = train_sarsa_lambda(
            baseline_env,
            episodes=ep,
            alpha=args.alpha,
            epsilon=args.epsilon,
            gamma=args.gamma,
            lambda_=args.lambda_,
            seed=args.seed,
        )

        plot_all_initial_q(ep, out, alpha=args.alpha, lambda_=args.lambda_, seed=args.seed, result=baseline_result)
        plot_win_rate_baseline(ep, out, alpha=args.alpha, lambda_=args.lambda_, seed=args.seed, result=baseline_result)
        plot_dice_comparison(ep, out, alpha=args.alpha, lambda_=args.lambda_, seed=args.seed, result=baseline_result)
 
        run_lambda = not args.skip_lambda_sweep
        run_alpha = not args.skip_alpha_sweep
        run_episode = not args.skip_episode_sweep
        episode_values = tuple(sorted({max(1_000, ep // 10), max(2_000, ep // 2), ep}))

        if run_lambda and run_alpha and args.parallel_sweeps:
            print(f"\n=== Running λ and α sweeps in parallel ({ep:,} episodes each) ===")
            with ProcessPoolExecutor(max_workers=2) as ex:
                future_lambda = ex.submit(plot_lambda_sweep, ep, out, alpha=args.alpha, seed=args.seed)
                future_alpha = ex.submit(plot_alpha_sweep, ep, out, lambda_=args.lambda_, seed=args.seed)
                future_lambda.result()
                future_alpha.result()
        else:
            if run_lambda:
                print(f"\n=== λ sweep ({ep:,} episodes each) ===")
                plot_lambda_sweep(ep, out, alpha=args.alpha, seed=args.seed)

            if run_alpha:
                print(f"\n=== α sweep ({ep:,} episodes each) ===")
                plot_alpha_sweep(ep, out, lambda_=args.lambda_, seed=args.seed)

        if run_episode:
            print(f"\n=== Episode sweep ({', '.join(f'{v:,}' for v in episode_values)} episodes) ===")
            plot_episode_sweep(
                out,
                episode_values,
                alpha=args.alpha,
                lambda_=args.lambda_,
                seed=args.seed,
            )
 
        print(f"\nDone — figures saved to {out}")
        return 0
 
    # ------------------------------------------------------------------ train
    env = RoyalGameOfUrEnv()
    result = train_sarsa_lambda(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        lambda_=args.lambda_,
        seed=args.seed,
    )
    rewards_figure, performance_figure, initial_values_figure = plot_training_result(result, args.output_dir)
 
    print(f"Finished {args.episodes} training episodes.")
    print(f"Saved reward plot to {rewards_figure}")
    print(f"Saved win-rate plot to {performance_figure}")
    print(f"Saved initial-value plot to {initial_values_figure}")
    return 0
