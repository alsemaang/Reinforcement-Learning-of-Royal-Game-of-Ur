# Reinforcement-Learning-of-Royal-Game-of-Ur
Implement and apply the algorithms to the Royal Game of Ur. 

# Basic training:

Default settings (1000 episodes)
uv run python -m royal_game_of_ur --episodes 10000 --output-dir artifacts

Custom hyperparameters
uv run python -m royal_game_of_ur --episodes 10000 --alpha 0.05 --lambda 0.9 --epsilon 0.1 --gamma 1.0 --seed 42 --output-dir artifacts

# Demo Run:
uv run python -m royal_game_of_ur --demo

# Analysis:

Full analysis (baseline + λ sweep + α sweep) — expect ~20 min at 100k
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts

Skip the expensive sweeps for a quick test
uv run python -m royal_game_of_ur --analyse --episodes 10000 --output-dir artifacts --skip-lambda-sweep --skip-alpha-sweep

Only skip α sweep
uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts --skip-alpha-sweep

Analysis with custom baseline hyperparameters
uv run python -m royal_game_of_ur --analyse --episodes 100000 --alpha 0.05 --lambda 0.9 --output-dir artifacts