from __future__ import annotations

from .environment import RoyalGameOfUrEnv


def main() -> int:
    env = RoyalGameOfUrEnv()
    observation, info = env.reset(seed=0)

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
