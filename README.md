# Reinforcement-Learning-of-Royal-Game-of-Ur
Implement and apply reinforcement learning algorithms to the Royal Game of Ur.

## Quick start

1) Install dependencies
uv sync

2) Run a short environment demo (random moves)
uv run python -m royal_game_of_ur --demo

3) Run default training (1000 episodes)
uv run python -m royal_game_of_ur --output-dir artifacts

This training command saves:
- sarsa_lambda_rewards.png
- sarsa_lambda_win_rate.png
- sarsa_lambda_initial_values.png

## Training commands

Default settings
uv run python -m royal_game_of_ur --output-dir artifacts

Custom hyperparameters
uv run python -m royal_game_of_ur --episodes 10000 --alpha 0.05 --lambda 0.9 --epsilon 0.1 --gamma 1.0 --seed 42 --output-dir artifacts/custom_train

## Analysis commands

Full analysis (baseline + lambda sweep + alpha sweep + episode sweep)
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_2

Full analysis with parallel lambda and alpha sweeps (faster)
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_2 --parallel-sweeps

Quick analysis smoke test (only baseline plots)
uv run python -m royal_game_of_ur --analyse --episodes 10000 --output-dir artifacts/smoke --skip-lambda-sweep --skip-alpha-sweep --skip-episode-sweep

Skip only episode sweep
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_2 --skip-episode-sweep

Skip only alpha sweep
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_2 --skip-alpha-sweep

Custom baseline hyperparameters for analysis
uv run python -m royal_game_of_ur --analyse --episodes 100000 --alpha 0.05 --lambda 0.9 --epsilon 0.1 --gamma 1.0 --seed 42 --output-dir artifacts/Run_custom

## Notes

- Without skip flags, all analysis sweeps run.
- Output directory is created automatically.
- Use a new output directory per run to keep old figures.