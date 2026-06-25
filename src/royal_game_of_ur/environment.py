from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from gymnasium import Env, spaces


Position: TypeAlias = int
Move: TypeAlias = dict[str, int]

START: Position = 0


@dataclass(frozen=True, slots=True)
class GameConfig:
    pieces_per_player: int = 2
    track_length: int = 14
    rosette_positions: tuple[int, ...] = (4, 8, 14)


class RoyalGameOfUrEnv(Env[dict[str, np.ndarray], Move]):
    """Gymnasium environment for the Royal Game of Ur.

    Player 1 is the learning agent. Player 2 acts randomly inside the
    environment until the next player 1 decision point.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        pieces_per_player: int = 2,
        track_length: int = 14,
        rosette_positions: tuple[int, ...] = (4, 8, 14),
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

        self.start_square = START
        self.end_square = self.config.track_length + 1

        self.action_space = spaces.Dict(
            {
                "start": spaces.Discrete(self.config.track_length + 2),
                "end": spaces.Discrete(self.config.track_length + 2),
            }
        )
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

        self._player_1 = np.full(self.config.pieces_per_player, START, dtype=np.int16)
        self._player_2 = np.full(self.config.pieces_per_player, START, dtype=np.int16)
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
        self._advance_to_player_1_turn()
        return self._observation(), self._info()

    def step(
        self, action: Move
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step() called on a terminated environment")
        if self._current_player != 1:
            self._advance_to_player_1_turn()

        move = self._normalize_action(action)
        legal_actions = self.legal_actions(1)
        real_legal_actions = [legal_action for legal_action in legal_actions if legal_action != self._pass_action()]
        if move == self._pass_action():
            if real_legal_actions:
                raise ValueError(f"Illegal action {move}; legal actions are {legal_actions}")
            self._current_player = 2
            self._current_dice = self._roll_dice()
            self._play_player_2_turns()
            return self._observation(), 0.0, self._terminated, False, self._info()

        if move not in legal_actions:
            raise ValueError(f"Illegal action {move}; legal actions are {legal_actions}")

        reward = 0.0
        self._apply_move(1, move)

        if self._has_won(1):
            self._terminated = True
            self._winner = 1
            reward = 1.0
        elif self._move_ended_on_rosette(move["end"]):
            self._current_player = 1
            self._current_dice = self._roll_dice()
        else:
            self._current_player = 2
            self._current_dice = self._roll_dice()
            self._play_player_2_turns()

        return self._observation(), reward, self._terminated, False, self._info()

    def render(self) -> None:
        print(self._render_text())

    def close(self) -> None:
        return None

    def legal_actions(self, player: int) -> list[Move]:
        pieces = self._player_1 if player == 1 else self._player_2
        opponent = self._player_2 if player == 1 else self._player_1
        legal: list[Move] = []
        seen: set[tuple[int, int]] = set()

        for piece_index, start in enumerate(pieces):
            start = int(start)
            if start == self.end_square:
                continue

            end = self._destination(start, self._current_dice)
            if end is None:
                continue
            if self._is_occupied_by_own_piece(end, pieces, piece_index):
                continue
            if self._is_occupied_by_opponent(end, opponent):
                continue
            move_key = (start, end)
            if move_key in seen:
                continue
            seen.add(move_key)
            legal.append({"start": start, "end": end})

        if not legal:
            return [self._pass_action()]
        return legal

    def sample_action(self) -> Move:
        legal = self.legal_actions(1)
        if not legal:
            return {"start": START, "end": START}
        index = int(self.np_random.integers(0, len(legal)))
        return legal[index]

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "player_1": self._player_1.copy(),
            "player_2": self._player_2.copy(),
            "dice": np.array(self._current_dice, dtype=np.int64),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "current_player": self._current_player,
            "winner": self._winner,
            "legal_actions": self.legal_actions(1) if not self._terminated else [],
        }

    def _normalize_action(self, action: Move) -> Move:
        if not isinstance(action, dict):
            raise TypeError("Action must be a dict with 'start' and 'end'")
        if "start" not in action or "end" not in action:
            raise ValueError("Action must contain 'start' and 'end'")
        return {"start": int(action["start"]), "end": int(action["end"])}

    def _pass_action(self) -> Move:
        return {"start": START, "end": START}

    def _roll_dice(self) -> int:
        return int(self.np_random.choice(np.arange(5), p=np.array([1, 4, 6, 4, 1]) / 16))

    def _destination(self, start: Position, dice: int) -> Position | None:
        if dice == 0:
            return None
        if start == self.end_square:
            return None

        end = start + dice
        if end > self.end_square:
            return None
        return end

    def _apply_move(self, player: int, move: Move) -> None:
        if move == self._pass_action():
            return

        pieces = self._player_1 if player == 1 else self._player_2
        opponent = self._player_2 if player == 1 else self._player_1

        piece_index = self._find_piece_at_position(pieces, move["start"])
        if piece_index is None:
            raise ValueError(f"No piece at start square {move['start']}")

        if move["end"] != self.end_square and move["end"] not in self.config.rosette_positions:
            opponent_piece = self._find_piece_at_position(opponent, move["end"])
            if opponent_piece is not None:
                opponent[opponent_piece] = START

        pieces[piece_index] = move["end"]

    def _play_player_2_turns(self) -> None:
        while not self._terminated and self._current_player == 2:
            legal_actions = self.legal_actions(2)
            if not legal_actions:
                self._current_player = 1
                self._current_dice = self._roll_dice()
                break

            action = legal_actions[int(self.np_random.integers(0, len(legal_actions)))]
            if action == self._pass_action():
                self._current_player = 1
                self._current_dice = self._roll_dice()
                break

            self._apply_move(2, action)

            if self._has_won(2):
                self._terminated = True
                self._winner = 2
                break

            if self._move_ended_on_rosette(action["end"]):
                self._current_player = 2
                self._current_dice = self._roll_dice()
            else:
                self._current_player = 1
                self._current_dice = self._roll_dice()
                break

    def _advance_to_player_1_turn(self) -> None:
        while not self._terminated and self._current_player != 1:
            self._play_player_2_turns()

    def _has_won(self, player: int) -> bool:
        pieces = self._player_1 if player == 1 else self._player_2
        return bool(np.all(pieces == self.end_square))

    def _move_ended_on_rosette(self, destination: int) -> bool:
        return destination in self.config.rosette_positions

    def _is_occupied_by_own_piece(
        self, destination: Position, pieces: np.ndarray, moved_piece_index: int
    ) -> bool:
        for index, position in enumerate(pieces):
            if index == moved_piece_index:
                continue
            if int(position) == destination:
                return True
        return False

    def _is_occupied_by_opponent(self, destination: Position, opponent: np.ndarray) -> bool:
        if destination == self.end_square:
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