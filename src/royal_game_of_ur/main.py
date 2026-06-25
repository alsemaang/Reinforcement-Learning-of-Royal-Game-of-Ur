from __future__ import annotations

from pathlib import Path
import argparse

from .environment import RoyalGameOfUrEnv
from .training import plot_training_result, train_sarsa_lambda


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
    args = parser.parse_args()

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
    rewards_figure, performance_figure = plot_training_result(result, args.output_dir)

    print(f"Finished {args.episodes} training episodes.")
    print(f"Saved reward plot to {rewards_figure}")
    print(f"Saved win-rate plot to {performance_figure}")
    return 0
