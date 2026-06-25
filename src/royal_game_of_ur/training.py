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


@dataclass(slots=True)
class TrainingResult:
    episode_rewards: list[float]
    wins: list[int]
    moving_average: list[float]
    tracked_initial_values: dict[int, list[float]]
    q_table: dict[tuple[tuple[int, ...], tuple[int, ...], int], dict[tuple[int, int], float]]


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
    rng = np.random.default_rng(seed)
    q_table: dict[tuple[tuple[int, ...], tuple[int, ...], int], dict[tuple[int, int], float]] = {}
    episode_rewards: list[float] = []
    wins: list[int] = []
    moving_average: list[float] = []
    tracked_initial_keys = _initial_state_action_keys(env)
    tracked_initial_values: dict[int, list[float]] = {dice: [] for dice in tracked_initial_keys}

    progress = tqdm(range(episodes), desc="Training episodes", unit="episode")
    for episode in progress:
        observation, _ = env.reset(seed=seed + episode)
        state = _state_key(observation)
        action = _epsilon_greedy_action(env, q_table, state, epsilon, rng)
        eligibility_traces: dict[tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, int]], float] = {}
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated and steps < max_steps_per_episode:
            next_observation, reward, terminated, truncated, _ = env.step(_action_dict(action))
            total_reward += reward
            next_state = _state_key(next_observation)
            steps += 1

            if steps >= max_steps_per_episode and not terminated:
                truncated = True

            if terminated or truncated:
                next_action = None
                td_target = reward
            else:
                next_action = _epsilon_greedy_action(env, q_table, next_state, epsilon, rng)
                td_target = reward + gamma * _q_value(q_table, next_state, next_action)

            current_q = _q_value(q_table, state, action)
            td_error = td_target - current_q

            trace_key = (state[0], state[1], state[2], action)
            eligibility_traces[trace_key] = eligibility_traces.get(trace_key, 0.0) + 1.0

            for trace_key, trace_value in list(eligibility_traces.items()):
                trace_state = (trace_key[0], trace_key[1], trace_key[2])
                trace_action = trace_key[3]
                q_table.setdefault(trace_state, {})
                q_table[trace_state][trace_action] = q_table[trace_state].get(trace_action, 0.0) + alpha * td_error * trace_value
                eligibility_traces[trace_key] = gamma * lambda_ * trace_value

            state = next_state
            if next_action is not None:
                action = next_action

        episode_rewards.append(total_reward)
        wins.append(1 if env._winner == 1 else 0)
        window = wins[max(0, len(wins) - evaluation_window) :]
        moving_average.append(float(np.mean(window)) if window else 0.0)

        progress.set_postfix(
            episode_reward=f"{total_reward:.1f}",
            win_rate=f"{moving_average[-1]:.2f}",
        )

        for dice_value, (state_key, action_key) in tracked_initial_keys.items():
            tracked_initial_values[dice_value].append(_q_value(q_table, state_key, action_key))

    return TrainingResult(
        episode_rewards=episode_rewards,
        wins=wins,
        moving_average=moving_average,
        tracked_initial_values=tracked_initial_values,
        q_table=q_table,
    )


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

    cumulative_win_rate = np.cumsum(wins) / episodes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, cumulative_win_rate, linewidth=2, color="tab:green")
    ax.set_title("Cumulative win rate of player 1")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(performance_figure, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for dice_value, values in sorted(result.tracked_initial_values.items()):
        ax.plot(episodes, values, linewidth=1.5, label=f"Initial dice {dice_value}")
    ax.set_title("Tracked Q-value of the five initial state-action pairs")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q(s, a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(initial_values_figure, dpi=200)
    plt.close(fig)

    return rewards_figure, performance_figure, initial_values_figure


def _state_key(observation: dict[str, np.ndarray]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    return (
        tuple(int(value) for value in observation["player_1"]),
        tuple(int(value) for value in observation["player_2"]),
        int(observation["dice"]),
    )


def _initial_state_action_keys(
    env: RoyalGameOfUrEnv,
) -> dict[int, tuple[tuple[tuple[int, ...], tuple[int, ...], int], tuple[int, int]]]:
    tracked_keys: dict[int, tuple[tuple[tuple[int, ...], tuple[int, ...], int], tuple[int, int]]] = {}
    empty_player_state = tuple(0 for _ in range(env.config.pieces_per_player))
    opponent_state = tuple(0 for _ in range(env.config.pieces_per_player))

    for dice_value in range(5):
        state_key = (empty_player_state, opponent_state, dice_value)
        if dice_value == 0:
            action_key = (0, 0)
        else:
            action_key = (0, dice_value)
        tracked_keys[dice_value] = (state_key, action_key)

    return tracked_keys


def _action_dict(action: tuple[int, int]) -> Move:
    return {"start": int(action[0]), "end": int(action[1])}


def _all_legal_actions(env: RoyalGameOfUrEnv) -> list[tuple[int, int]]:
    legal_actions = env.legal_actions(1)
    return [(int(move["start"]), int(move["end"])) for move in legal_actions]


def _q_value(
    q_table: dict[tuple[tuple[int, ...], tuple[int, ...], int], dict[tuple[int, int], float]],
    state: tuple[tuple[int, ...], tuple[int, ...], int],
    action: tuple[int, int],
) -> float:
    return q_table.get(state, {}).get(action, 0.0)


def _epsilon_greedy_action(
    env: RoyalGameOfUrEnv,
    q_table: dict[tuple[tuple[int, ...], tuple[int, ...], int], dict[tuple[int, int], float]],
    state: tuple[tuple[int, ...], tuple[int, ...], int],
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    legal_actions = _all_legal_actions(env)
    if not legal_actions:
        return (0, 0)

    if rng.random() < epsilon:
        index = int(rng.integers(0, len(legal_actions)))
        return legal_actions[index]

    state_values = [q_table.get(state, {}).get(action, 0.0) for action in legal_actions]
    best_value = max(state_values)
    best_actions = [action for action, value in zip(legal_actions, state_values) if value == best_value]
    index = int(rng.integers(0, len(best_actions)))
    return best_actions[index]