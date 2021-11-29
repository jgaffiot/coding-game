"""CodingGames challenge: Code Busters."""

import sys
from math import pi, sqrt, atan2, cos, sin
from typing import Dict, Optional, Union

X_MAX = 16000
Y_MAX = 9000
R_FOG = 2200
R_BUST_MAX = 1760
R_BUST_MIN = 900
R_RELEASE = 1600
GHOST_MOVE = 400
BUSTER_MOVE = 800
STUNNED_COOL_DOWN = 20
STUN_GUN_COOL_DOWN = 20
STUN_TIME = 40
R_START = R_RELEASE + BUSTER_MOVE

NB_BUSTERS = int(input())  # the amount of busters you control (2 ≤ NB_BUSTERS ≤ 5)
NB_GHOSTS = int(input())  # the amount of ghosts on the map (8 ≤ ghostCount ≤ 28)
MY_ID = int(input())  # 0: top left, 1: bottom right
MY_DIR = MY_ID * pi * 0.5
SIGN = 1 if MY_ID == 0 else -1


def debug(string: str) -> None:
    """Print to stderr to debug and avoid conflict with instruction printing."""
    print(string, file=sys.stderr, flush=True)


class Point:
    """A 2 dimension point."""

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        """Initialize self with cartesian coordinates."""
        self.x = int(x)
        self.y = int(y)

    @property
    def r(self):
        """Modulus."""
        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def theta(self):
        """Argument."""
        return atan2(self.y, self.x)

    def dist_to(self, p: "Point") -> float:
        """Distance to another point"""
        return sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    @classmethod
    def from_polar(cls, r, theta) -> "Point":
        """Create a Point from polar coordinates."""
        return cls(r * cos(theta), r * sin(theta))

    def advance(self, dist: int) -> None:
        """Advance the point from the given dist (same argument)."""
        theta = self.theta
        self.x += dist * cos(theta)
        self.y += dist * sin(theta)
        self.x = min(max(0, self.x), X_MAX)
        self.y = min(max(0, self.y), Y_MAX)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, other) -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __mul__(self, scalar) -> "Point":
        return Point(self.x * scalar, self.y * scalar)


def distance(a: Point, b: Point) -> float:
    """Return the distance between 2 Points."""
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


BASES = {0: Point(0, 0), 1: Point(16000, 9000)}
HOME: Point = BASES[MY_ID]
OPP: Point = BASES[(MY_ID + 1) % 2]
SQRT2 = int(sqrt(2) / 2.0) - 1

PATH = {
    0: [
        Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        # Point(HOME.x, OPP.y) - Point(1, -1) * (SQRT2 * R_FOG * SIGN),
        # Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * 3.0 * R_FOG),
        Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * 3.0 * R_FOG),
    ],
    1: [
        Point(OPP.x - SIGN * R_FOG, HOME.y + SIGN * R_FOG),
        # Point(OPP.x, HOME.y) - Point(-1, 1) * (SQRT2 * R_FOG * SIGN),
        # Point(OPP.x - SIGN * R_FOG, HOME.y + SIGN * R_FOG),
        Point(OPP.x - SIGN * R_FOG, HOME.y + SIGN * 3.0 * R_FOG),
        Point(HOME.x + SIGN * 2.0 * R_FOG, HOME.y + SIGN * 3.0 * R_FOG),
        Point(HOME.x + SIGN * 2.0 * R_FOG, HOME.y + SIGN * R_FOG),
    ],
    2: [
        Point(HOME.x + SIGN * Y_MAX / 2.0, HOME.y + SIGN * Y_MAX / 2.0),
        Point(OPP.x - SIGN * R_FOG / 2.0, HOME.y + SIGN * Y_MAX / 2.0),
    ],
    3: [
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(HOME.x + SIGN * R_FOG, HOME.y + SIGN * R_FOG),
    ],
    4: [
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(HOME.x + SIGN * R_FOG, HOME.y + SIGN * R_FOG),
    ],
}


class _Entity:
    """An entity on the play field."""

    def __init__(self):
        """Initialize self."""
        self.id: int = -1
        self.p: Point = Point(-1, -1)

    def update(self, id_: int, x: int, y: int) -> None:
        """Update the current entity with new data."""
        self.id = id_
        self.p = Point(x, y)

    def __repr__(self):
        return f"Entity({self.id}, {self.p.x}, {self.p.y})"

    def __str__(self):
        return f"Entity({self.id}, {self.p})"

    def dist_to(self, p: Point) -> float:
        """Return the distance to a given Point."""
        return self.p.dist_to(p)


class Ghost(_Entity):
    """A Ghost"""

    def __init__(self):
        """Initialize self."""
        super().__init__()
        self.pv: int = -1
        self.nb_buster: int = -1

    @classmethod
    def from_data(cls, id_: int, x: int, y: int, pv: int, nb_bust: int) -> "Ghost":
        """Return a new Ghost fully initialized with the given data."""
        new_ghost = cls()
        new_ghost.update(id_, x, y, pv, nb_bust)
        return new_ghost

    # noinspection PyMethodOverriding
    def update(self, id_: int, x: int, y: int, pv: int, nb_buster: int) -> None:
        """Update the current entity with new data."""
        super().update(id_, x, y)
        self.pv = pv
        self.nb_buster = nb_buster

    def __repr__(self):
        return f"Ghost({self.id}, {self.p.x}, {self.p.y})"

    def __str__(self):
        return f"Ghost {self.id}, {self.p}, nb={self.nb_buster}"


class TargetGhost(Ghost):
    """A targeted ghost."""

    def __init__(self, ghost: Ghost, buster_pos: Point):
        super().__init__()
        self.update(ghost.id, ghost.p.x, ghost.p.y, ghost.pv, ghost.nb_buster)
        self.d = buster_pos.dist_to(ghost.p)


class _Buster(_Entity):
    """A buster, mine or opponent."""

    def __init__(self):
        super().__init__()
        self.loaded: bool = False
        self.carried_ghost_id: int = 0

    # noinspection PyMethodOverriding
    def update(self, id_: int, x: int, y: int, loaded: int, ghost_id: int) -> None:
        """Update the current buster with new data."""
        super().update(id_, x, y)
        self.loaded: bool = loaded == 1
        self.carried_ghost_id: int = ghost_id

    def __repr__(self) -> str:
        return (
            f"Buster({self.id}, {self.p.x}, {self.p.y}, {self.loaded},"
            f" {self.carried_ghost_id})"
        )

    def __str__(self) -> str:
        return f"Buster {self.id}, {self.p}" + " loaded" if self.loaded else ""


class Opponent(_Buster):
    """One of the opponent busters."""

    def __init__(self):
        """Initialize self."""
        super().__init__()
        self.is_stunned = False
        self.is_visible = False
        self.last_stunned: int = -100

    def update(self, id_: int, x: int, y: int, loaded: int, ghost_id: int) -> None:
        """Update the current buster with new data."""
        super().update(id_, x, y, loaded, ghost_id)
        self.is_visible = True

    def stunned_at(self, turn_index: int) -> None:
        """Set to "is_stunned" state."""
        self.is_stunned = True
        self.last_stunned = turn_index


class TargetOpp(Opponent):
    """A targeted opponent buster."""

    def __init__(self, opp: Opponent, buster_pos: Point):
        super().__init__()
        self.update(opp.id, opp.p.x, opp.p.y, opp.loaded, opp.carried_ghost_id)
        self.is_stunned = opp.is_stunned
        self.d = buster_pos.dist_to(opp.p)


class Mine(_Buster):
    """One of my busters."""

    _current_number = 0

    def __init__(self):
        """Initialize self."""
        super().__init__()
        self.num = Mine._current_number
        Mine._current_number += 1
        self.last_stun: int = -100
        self._point_index: int = 0
        self.targets: Dict[int, TargetGhost] = {}
        self.opponents: Dict[int, TargetOpp] = {}
        self._is_exploring: bool = False

    @property
    def is_exploring(self) -> bool:
        """Return True if the buster is exploring"""
        return self._is_exploring

    @is_exploring.setter
    def is_exploring(self, val: bool) -> None:
        """Set the exploring property."""
        if val is False:
            self._point_index = 0
        self._is_exploring = val

    def compute_dist_to_ghost(self, ghosts: Dict[int, Ghost]) -> None:
        """Compute the distance to the given ghosts."""
        self.targets = {
            id_: TargetGhost(ghost, self.p) for id_, ghost in ghosts.items()
        }

    def compute_dist_to_opponents(self, opponents: Dict[int, Opponent]) -> None:
        """Compute the distance to the given ghosts."""
        self.opponents = {
            id_: TargetOpp(opp, self.p)
            for id_, opp in opponents.items()
            if opp.is_visible
        }

    def can_bust(self) -> Optional[int]:
        """Return the id of a ghost if it can be busted, else None."""
        for id_, target in self.targets.items():
            if R_BUST_MIN <= target.d <= R_BUST_MAX:
                return id_
        return None

    def can_stun(self) -> Optional[int]:
        """Return the id of an opponent if it can be stun, else None."""
        for id_, opp in self.opponents.items():
            if opp.d <= R_BUST_MAX and not opp.is_stunned:
                return id_
        return None

    def has_too_close_target(self) -> Optional[int]:
        """Return the id of a too close ghost, else None."""
        for id_, target in self.targets.items():
            if target.d <= R_BUST_MIN:
                return id_
        return None

    def compute_bust_point(self, ghost_id) -> Point:
        """Compute the point to bust a target."""
        target = self.targets[ghost_id]
        new = target.p
        new.advance(GHOST_MOVE)
        direction = new - self.p
        dist = self.p.dist_to(new)

        if dist >= R_BUST_MAX:
            return self.p + Point.from_polar(BUSTER_MOVE, direction.theta)
        if dist <= R_BUST_MIN:
            return self.p + Point.from_polar(BUSTER_MOVE, -direction.theta)
        return self.p

    def get_closest_target(self) -> Optional[TargetGhost]:
        """Get the closest target if any."""
        targets = [t for t in self.targets.values() if t.pv]
        if targets:
            return sorted(targets, key=lambda t: t.d)[0]
        return None

    def explore(self) -> Point:
        """Return a Point allowing to explore the map if no ghost is visible."""
        debug(f"{self.num} explores")
        path = PATH[self.num]

        if not self.is_exploring:
            self.is_exploring = True

        next_point = path[self._point_index]
        dist = self.dist_to(next_point)
        debug(
            f"next point: {next_point} ({self._point_index})" f" from {self.p} ({dist})"
        )
        if dist < 2:
            self._point_index += 1
            if self._point_index == len(path):
                self._point_index = 0
            next_point = path[self._point_index]
            debug(f"new point {next_point}")
        return next_point


def game_loop():
    """The game loop."""

    ghost_registry: Dict[int, Ghost] = {}
    mine_registry: Dict[int, Mine] = {}
    opponent_registry: Dict[int, Opponent] = {}

    turn_index = 0
    while True:
        turn_index += 1

        # Load the visible entities
        entities = int(input())  # the number of busters and ghosts visible to you
        visible_ghosts = {}
        for opponent in opponent_registry.values():
            opponent.is_visible = False
            if opponent.last_stunned > turn_index + STUNNED_COOL_DOWN:
                opponent.is_stunned = False

        for i in range(entities):
            # entity_id: buster id or ghost id
            # y: position of this buster / ghost
            # entity: the team id if it is a buster, -1 if it is a ghost.
            # state: For busters: 0=idle, 1=carrying a ghost.
            #        For ghosts: point of life
            # value: For busters: Ghost id being carried.
            #        For ghosts: number of busters attempting to trap this ghost.
            entity_id, x, y, entity, state, value = (int(j) for j in input().split())
            if entity == MY_ID:
                mine_registry.setdefault(entity_id, Mine()).update(
                    entity_id, x, y, state, value
                )
            elif entity == -1:
                visible_ghosts[entity_id] = Ghost.from_data(
                    entity_id, x, y, state, value
                )
            else:
                opponent_registry.setdefault(entity_id, Opponent()).update(
                    entity_id, x, y, state, value
                )

        for mine in mine_registry.values():
            if mine.loaded and mine.carried_ghost_id in ghost_registry:
                del ghost_registry[mine.carried_ghost_id]

        for opp in opponent_registry.values():
            if opp.loaded and opp.carried_ghost_id in ghost_registry:
                del ghost_registry[opp.carried_ghost_id]

        # Merge the visible ghosts into the registry
        invisible_ghosts = {
            k: v for k, v in ghost_registry.items() if k not in visible_ghosts
        }
        ghost_registry.update(visible_ghosts)
        debug(", ".join([str(i) for i in visible_ghosts.keys()]))
        debug(", ".join([str(i) for i in ghost_registry.keys()]))

        for id_, buster in mine_registry.items():
            buster: Mine
            # Try to stun
            if turn_index - buster.last_stun >= STUN_GUN_COOL_DOWN:
                buster.compute_dist_to_opponents(opponent_registry)
                target_id = buster.can_stun()
                if target_id is not None:
                    print(f"STUN {target_id}")
                    buster.last_stun = turn_index
                    opponent_registry[target_id].stunned_at(turn_index)
                    continue

            # Go back home if loaded
            if buster.loaded:
                if buster.dist_to(HOME) <= R_RELEASE:
                    print("RELEASE")
                    continue
                print(f"MOVE {HOME.x} {HOME.y}")
                continue

            if not NB_GHOSTS:
                print(f"MOVE {OPP.x} {OPP.y}")

            # No visible nor invisible ghosts, so exploring
            if not [g for g in visible_ghosts.values() if g.pv] and not [
                g for g in invisible_ghosts.values() if g.pv
            ]:
                p = buster.explore()
                print(f"MOVE {p.x} {p.y}")
                continue

            # No visible but invisible ghosts
            buster.is_exploring = False
            if not [g for g in visible_ghosts.values() if g.pv]:
                buster.compute_dist_to_ghost(invisible_ghosts)
                target = buster.get_closest_target()
                ghost_registry[target.id].pv -= 1
                debug(f"{buster.num} targets INvisible {target.id}")
                print(f"MOVE {target.p.x} {target.p.y}")
                continue

            # Visible ghosts
            buster.compute_dist_to_ghost(visible_ghosts)
            # for target_id, t in buster.targets.items():
            #     debug(f"{target_id} -> {t.d}")

            target_id = buster.can_bust()
            if target_id is not None:
                print(f"BUST {target_id}")
                ghost_registry[target_id].pv -= 1
                continue

            target_id = buster.has_too_close_target()
            if target_id is not None:
                p = buster.compute_bust_point(target_id)
                print(f"MOVE {p.x} {p.y}")
                ghost_registry[target_id].pv -= 1
                continue

            target = buster.get_closest_target()
            if target is not None:
                ghost_registry[target.id].pv -= 1
                debug(f"{buster.num} targets visible {target.id}")
                print(f"MOVE {target.p.x} {target.p.y}")
                continue

            debug(f"ERROR: VISIBLE GHOST BUT NO TARGET FOR: {buster}")
            print(f"MOVE {X_MAX/2} {Y_MAX/2}")


if __name__ == "__main__":
    game_loop()
