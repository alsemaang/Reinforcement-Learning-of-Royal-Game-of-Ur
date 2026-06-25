from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from gymnasium import Env, spaces


Position: TypeAlias = int

START: Position = 0


@dataclass(frozen=True, slots=True)
class GameConfig:
    pieces_per_player: int = 2
    track_length: int = 14
    rosette_positions: tuple[int, ...] = (4, 8, 14)


class RoyalGameOfUrEnv(Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for the Royal Game of Ur.

    This follows the single-agent option from the assignment: the learning agent
    plays player 1, while player 2 is resolved inside the environment with a
    random policy until the next player 1 decision point.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        pieces_per_player: int = 2,
        track_length: int = 14,
        rosette_positions: tuple[int, ...] = (4, 8, 14),
        render_mode: Literal["human"] | None = None,
    ) -> None:
        super().__init__()
        if pieces_per_player < 1:
            raise ValueError("pieces_per_player must be at least 1")
        if track_length < 2:
            raise ValueError("track_length must be at least 2")

        self.config = GameConfig(
            pieces_per_player=pieces_per_player,
            track_length=track_length,
            rosette_positions=tuple(rosette_positions),
        )
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.config.pieces_per_player + 1)
        self.observation_space = spaces.Dict(
            {
                "player_1": spaces.MultiDiscrete(
                    [self.config.track_length + 2] * self.config.pieces_per_player
                ),
                "player_2": spaces.MultiDiscrete(
                    [self.config.track_length + 2] * self.config.pieces_per_player
                ),
                "dice": spaces.Discrete(5),
            }
        )

        self.np_random = np.random.default_rng()
        self._player_1 = np.zeros(self.config.pieces_per_player, dtype=np.int16)
        self._player_2 = np.zeros(self.config.pieces_per_player, dtype=np.int16)
        self._current_player = 1
        self._current_dice = 0
        self._terminated = False
        self._winner: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._player_1.fill(START)
        self._player_2.fill(START)
        self._current_player = 1
        self._current_dice = self._roll_dice()
        self._terminated = False
        self._winner = None
        self._resolve_opponent_turns()
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step() called on a terminated environment")

        reward = 0.0

        if self._current_player != 1:
            self._resolve_opponent_turns()

        legal_moves = self.legal_actions(1)
        if action not in legal_moves:
            reward = -0.1
            self._current_player = 2
        elif action != self.config.pieces_per_player:
            self._move_piece(1, action, self._current_dice)
            if self._has_won(1):
                self._terminated = True
                self._winner = 1
                reward = 1.0
            elif self._moved_piece_is_rosette(1, action):
                self._current_player = 1
                self._current_dice = self._roll_dice()
            else:
                self._current_player = 2
        else:
            self._current_player = 2

        if not self._terminated and self._current_player == 2:
            self._resolve_opponent_turns()

        return self._observation(), reward, self._terminated, False, self._info()

    def render(self) -> None:
        print(self._render_text())

    def close(self) -> None:
        return None

    def legal_actions(self, player: int) -> list[int]:
        dice = self._current_dice
        pieces = self._player_1 if player == 1 else self._player_2
        opponent = self._player_2 if player == 1 else self._player_1
        legal: list[int] = []

        for piece_index, position in enumerate(pieces):
            destination = self._destination(int(position), dice)
            if destination is None:
                continue
            if self._is_occupied_by_own_piece(destination, pieces, piece_index):
                continue
            if self._is_occupied_by_opponent(destination, opponent):
                if destination in self.config.rosette_positions:
                    continue
            legal.append(piece_index)

        legal.append(self.config.pieces_per_player)
        return legal

    def sample_action(self) -> int:
        legal = self.legal_actions(1)
        return int(self.np_random.choice(legal))

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "player_1": self._encode_positions(self._player_1),
            "player_2": self._encode_positions(self._player_2),
            "dice": np.array(self._current_dice, dtype=np.int64),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "current_player": self._current_player,
            "winner": self._winner,
            "legal_actions": self.legal_actions(1) if not self._terminated else [],
        }

    def _encode_positions(self, positions: np.ndarray) -> np.ndarray:
        encoded = positions.copy()
        encoded[encoded == START] = 0
        encoded[encoded == self.config.track_length + 1] = self.config.track_length + 1
        return encoded

    def _roll_dice(self) -> int:
        return int(self.np_random.choice(np.arange(5), p=np.array([1, 4, 6, 4, 1]) / 16))

    def _destination(self, position: Position, dice: int) -> Position | None:
        if dice == 0:
            return None
        if position == self.config.track_length + 1:
            return None
        destination = position + dice
        if destination > self.config.track_length + 1:
            return None
        return destination

    def _move_piece(self, player: int, piece_index: int, dice: int) -> None:
        pieces = self._player_1 if player == 1 else self._player_2
        opponent = self._player_2 if player == 1 else self._player_1

        destination = self._destination(int(pieces[piece_index]), dice)
        if destination is None:
            return

        if self._is_occupied_by_opponent(destination, opponent):
            opponent_piece = self._find_piece_at_position(opponent, destination)
            if opponent_piece is not None:
                opponent[opponent_piece] = START

        pieces[piece_index] = destination

    def _resolve_opponent_turns(self) -> None:
        while not self._terminated and self._current_player == 2:
            self._current_dice = self._roll_dice()
            legal_moves = self.legal_actions(2)
            movable = [move for move in legal_moves if move != self.config.pieces_per_player]
            if not movable or self._current_dice == 0:
                self._current_player = 1
                self._current_dice = self._roll_dice()
                continue

            action = int(self.np_random.choice(movable))
            self._move_piece(2, action, self._current_dice)
            if self._has_won(2):
                self._terminated = True
                self._winner = 2
                return

            if self._moved_piece_is_rosette(2, action):
                self._current_player = 2
            else:
                self._current_player = 1
                self._current_dice = self._roll_dice()

    def _has_won(self, player: int) -> bool:
        pieces = self._player_1 if player == 1 else self._player_2
        return bool(np.all(pieces == self.config.track_length + 1))

    def _moved_piece_is_rosette(self, player: int, piece_index: int) -> bool:
        pieces = self._player_1 if player == 1 else self._player_2
        return int(pieces[piece_index]) in self.config.rosette_positions

    def _is_occupied_by_own_piece(
        self, destination: Position, pieces: np.ndarray, moved_piece_index: int
    ) -> bool:
        if destination == self.config.track_length + 1:
            return False
        for index, position in enumerate(pieces):
            if index == moved_piece_index:
                continue
            if int(position) == destination:
                return True
        return False

    def _is_occupied_by_opponent(self, destination: Position, opponent: np.ndarray) -> bool:
        if destination == self.config.track_length + 1:
            return False
        if destination in self.config.rosette_positions:
            return False
        return self._find_piece_at_position(opponent, destination) is not None

    def _find_piece_at_position(
        self, pieces: np.ndarray, position: Position
    ) -> int | None:
        matches = np.where(pieces == position)[0]
        if len(matches) == 0:
            return None
        return int(matches[0])

    def _render_text(self) -> str:
        return (
            f"P1={self._player_1.tolist()} P2={self._player_2.tolist()} "
            f"dice={self._current_dice} current_player={self._current_player} "
            f"winner={self._winner}"
        )


def make_env(**kwargs: Any) -> RoyalGameOfUrEnv:
    return RoyalGameOfUrEnv(**kwargs)