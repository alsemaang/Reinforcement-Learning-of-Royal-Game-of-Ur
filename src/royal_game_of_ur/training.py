"""Training utilities for SARSA(lambda).

This module is called by the main CLI. Run training via:
    uv run python -m royal_game_of_ur --episodes 1000 --output-dir artifacts

Run analysis (which reuses training internally) via:
    uv run python -m royal_game_of_ur --analyse --episodes 100000 --output-dir artifacts/Run_3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .environment import Move, RoyalGameOfUrEnv


StateKey = tuple[tuple[int, ...], tuple[int, ...], int]
ActionKey = tuple[int, int]
TraceKey = tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, int]]

QTable = dict[StateKey, dict[ActionKey, float]]
EligibilityTraces = dict[TraceKey, float]


@dataclass(slots=True)
class TrainingResult:
    episode_rewards: list[float]
    wins: list[int]
    moving_average: list[float]
    tracked_initial_values: dict[int, list[float]]
    q_table: QTable


def train_sarsa_lambda(
    env: RoyalGameOfUrEnv,
    episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    lambda_: float = 0.8,
    seed: int = 0,
    evaluation_window: int = 50,
    max_steps_per_episode: int = 500,
) -> TrainingResult:
    """Train an agent on the Royal Game of Ur using SARSA(λ) with
    replacing eligibility traces.

    The reward signal follows the MDP specification: r(s, a, s') = 1 when
    s' is the terminal winning state for player 1, and 0 otherwise.
    Only win (+1) is observed; loss is absorbing with reward 0, so the
    agent learns to maximise win probability.
    
    """
    rng = np.random.default_rng(seed)
    q_table: QTable = {}
    episode_rewards: list[float] = []
    wins: list[int] = []
    moving_average: list[float] = []
    tracked_initial_keys = _initial_state_action_keys(env)
    tracked_initial_values: dict[int, list[float]] = {d: [] for d in tracked_initial_keys}

    progress = tqdm(range(episodes), desc="Training episodes", unit="episode")

    for episode in progress:
        observation, info = env.reset(seed=seed + episode)
        state = _state_key(observation)
        legal = _legal_from_info(info)
        action = _epsilon_greedy(q_table, state, legal, epsilon, rng)

        eligibility_traces: EligibilityTraces = {}
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, next_info = env.step(
                _action_dict(action)
            )
            total_reward += reward
            steps += 1

            if steps >= max_steps_per_episode and not terminated:
                truncated = True

            next_state = _state_key(next_observation)

            # --- SARSA(λ) target -------------------------------------------------
            if terminated or truncated:
                # Terminal / timeout: no next action, bootstrap value is 0
                td_target = reward
                next_action: ActionKey | None = None
            else:
                next_legal = _legal_from_info(next_info)
                next_action = _epsilon_greedy(q_table, next_state, next_legal, epsilon, rng)
                td_target = reward + gamma * _q(q_table, next_state, next_action)

            td_error = td_target - _q(q_table, state, action)

            # --- Update eligibility traces (replacing traces) --------------------
            # Increment the current (s, a) trace first — *before* the decay loop
            # so that the Q-update below uses the correct, non-decayed value.
            trace_key: TraceKey = (*state, action)  # type: ignore[assignment]
            eligibility_traces[trace_key] = 1.0  # replacing trace: clamp to 1

            # --- Q-update for all traced state-action pairs ----------------------
            new_traces: EligibilityTraces = {}
            for tk, e in eligibility_traces.items():
                ts: StateKey = (tk[0], tk[1], tk[2])
                ta: ActionKey = tk[3]
                q_table.setdefault(ts, {})
                q_table[ts][ta] = q_table[ts].get(ta, 0.0) + alpha * td_error * e
                decayed = gamma * lambda_ * e
                if abs(decayed) > 1e-9:          # prune negligible traces
                    new_traces[tk] = decayed
            eligibility_traces = new_traces

            state = next_state
            if next_action is not None:
                action = next_action

        # --- Episode bookkeeping ------------------------------------------------
        episode_rewards.append(total_reward)
        won = 1 if env._winner == 1 else 0
        wins.append(won)
        window = wins[max(0, len(wins) - evaluation_window):]
        moving_average.append(float(np.mean(window)))

        progress.set_postfix(
            reward=f"{total_reward:.1f}",
            win_rate=f"{moving_average[-1]:.2f}",
        )

        for dice_value, (sk, ak) in tracked_initial_keys.items():
            tracked_initial_values[dice_value].append(_q(q_table, sk, ak))

    return TrainingResult(
        episode_rewards=episode_rewards,
        wins=wins,
        moving_average=moving_average,
        tracked_initial_values=tracked_initial_values,
        q_table=q_table,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_result(result: TrainingResult, output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rewards_figure = output_path / "sarsa_lambda_rewards.png"
    performance_figure = output_path / "sarsa_lambda_win_rate.png"
    initial_values_figure = output_path / "sarsa_lambda_initial_values.png"

    episodes = np.arange(1, len(result.episode_rewards) + 1)
    reward_array = np.asarray(result.episode_rewards, dtype=float)
    moving_average = np.asarray(result.moving_average, dtype=float)
    wins = np.asarray(result.wins, dtype=float)

    plt.style.use("ggplot")

    # Reward + moving average win rate
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, reward_array, alpha=0.3, linewidth=1, label="Episode reward")
    ax.plot(episodes, moving_average, linewidth=2, label="Moving average win rate")
    ax.set_title(r"SARSA($\lambda$) training on Royal Game of Ur")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / win rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rewards_figure, dpi=200)
    plt.close(fig)

    # Cumulative win rate
    cumulative_win_rate = np.cumsum(wins) / episodes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, cumulative_win_rate, linewidth=2, color="tab:green")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_title("Cumulative win rate of player 1")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(performance_figure, dpi=200)
    plt.close(fig)

    # Tracked Q-values for the initial state
    fig, ax = plt.subplots(figsize=(10, 5))
    for dice_value, values in sorted(result.tracked_initial_values.items()):
        label = f"dice={dice_value} (pass)" if dice_value == 0 else f"dice={dice_value}"
        ax.plot(episodes, values, linewidth=1.5, label=label)
    ax.set_title("Q-value of initial state-action pairs over training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q(s, a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(initial_values_figure, dpi=200)
    plt.close(fig)

    return rewards_figure, performance_figure, initial_values_figure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _state_key(observation: dict[str, np.ndarray]) -> StateKey:
    return (
        tuple(int(v) for v in observation["player_1"]),
        tuple(int(v) for v in observation["player_2"]),
        int(observation["dice"]),
    )


def _legal_from_info(info: dict[str, Any]) -> list[ActionKey]:
    """Convert the legal_actions list from env info into ActionKey tuples."""
    return [(int(m["start"]), int(m["end"])) for m in info["legal_actions"]]


def _action_dict(action: ActionKey) -> Move:
    return {"start": int(action[0]), "end": int(action[1])}


def _q(q_table: QTable, state: StateKey, action: ActionKey) -> float:
    return q_table.get(state, {}).get(action, 0.0)


def _epsilon_greedy(
    q_table: QTable,
    state: StateKey,
    legal_actions: list[ActionKey],
    epsilon: float,
    rng: np.random.Generator,
) -> ActionKey:
    """Select an action using ε-greedy policy over the provided legal actions."""
    if not legal_actions:
        # Should not happen in a well-formed episode, but guard anyway
        return (0, 0)

    if rng.random() < epsilon:
        return legal_actions[int(rng.integers(0, len(legal_actions)))]

    state_q = q_table.get(state, {})
    values = [state_q.get(a, 0.0) for a in legal_actions]
    best_val = max(values)
    best = [a for a, v in zip(legal_actions, values) if v == best_val]
    return best[int(rng.integers(0, len(best)))]


def _initial_state_action_keys(
    env: RoyalGameOfUrEnv,
) -> dict[int, tuple[StateKey, ActionKey]]:
    """Return one representative (state, action) pair per dice value,
    all starting from the all-zeros board state.

    For dice=0 the only legal action is the pass (0→0).
    For dice=d>0 the first piece moves from square 0 to square d.
    """
    n = env.config.pieces_per_player
    empty: tuple[int, ...] = tuple(0 for _ in range(n))
    tracked: dict[int, tuple[StateKey, ActionKey]] = {}

    for dice_value in range(5):
        state_key: StateKey = (empty, empty, dice_value)
        action_key: ActionKey = (0, 0) if dice_value == 0 else (0, dice_value)
        tracked[dice_value] = (state_key, action_key)

    return tracked