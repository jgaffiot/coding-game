"""CodingGames challenge: Code Busters."""

import sys
from copy import deepcopy
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
COOLDOWN = 20
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


def dist(a: Point, b: Point) -> float:
    """Return the distance between 2 Points."""
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


BASES = {0: Point(0, 0), 1: Point(16000, 9000)}
HOME: Point = BASES[MY_ID]
OPP: Point = BASES[(MY_ID + 1) % 2]

PATH = {
    0: [
        Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(HOME.x, OPP.y) + Point(1, -1) * (sqrt(2) / 2.0 * R_FOG * SIGN),
        Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * R_FOG),
        Point(OPP.x - SIGN * R_FOG, OPP.y - SIGN * 3.0 * R_FOG),
        Point(HOME.x + SIGN * R_FOG, OPP.y - SIGN * 3.0 * R_FOG),
    ],
    1: [
        Point(OPP.x - SIGN * R_FOG, HOME.y + SIGN * R_FOG),
        Point(OPP.x, HOME.y) + Point(-1, 1) * (sqrt(2) / 2.0 * R_FOG * SIGN),
        Point(OPP.x - SIGN * R_FOG, HOME.y + SIGN * R_FOG),
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

point_index: Dict[int, int] = {}
stun_opponents: Dict[int, int] = {}


class Entity:
    """An entity on the play field."""

    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.p = Point(x, y)

    def __repr__(self):
        return f"Entity({self.id}, {self.p.x}, {self.p.y})"

    def __str__(self):
        return f"Entity({self.id}, {self.p})"

    def dist_to(self, p: Point) -> float:
        """Return the distance to a given Point."""
        return self.p.dist_to(p)


class Ghost(Entity):
    """A Ghost"""

    def __init__(self, id_: int, x: int, y: int, nb_bust: int):
        super().__init__(id_, x, y)
        self.nb_bust = nb_bust

    def __repr__(self):
        return f"Ghost({self.id}, {self.p.x}, {self.p.y})"

    def __str__(self):
        return f"Ghost {self.id}, {self.p}, nb={self.nb_bust}"


class Target(Entity):
    """A targeted ghost or opponent."""

    def __init__(self, entity: Entity, buster_pos: Point):
        super().__init__(entity.id, entity.p.x, entity.p.y)
        self.d = buster_pos.dist_to(entity.p)


class Buster(Entity):
    """A buster, mine or opponent."""

    def __init__(self, id_: int, x: int, y: int, loaded: int, ghost_id: int):
        super().__init__(id_, x, y)
        self.loaded: bool = bool(loaded)
        self.carried_ghost_id: int = ghost_id
        self.targets: Dict[int, Target] = {}
        self.opponents: Dict[int, Target] = {}
        self._is_exploring: bool = False

    def __repr__(self):
        return (
            f"Buster({self.id}, {self.p.x}, {self.p.y}, {self.loaded},"
            f" {self.carried_ghost_id})"
        )

    def __str__(self):
        return f"Buster {self.id}, {self.p}" + " loaded" if self.loaded else ""

    @property
    def is_exploring(self) -> bool:
        """Return True if the buster is exploring"""
        return self._is_exploring

    @is_exploring.setter
    def is_exploring(self, val: bool) -> None:
        """Set the exploring property."""
        if val is False:
            point_index[self.id] = 0
        self._is_exploring = val

    def compute_dist_to_ghost(self, ghosts: Dict[int, Ghost]) -> None:
        """Compute the distance to the given ghosts."""
        for id_, ghost in ghosts.items():
            self.targets[id_] = Target(ghost, self.p)

    def compute_dist_to_opponents(self, opponents: Dict[int, "Buster"]) -> None:
        """Compute the distance to the given ghosts."""
        for id_, opp in opponents.items():
            self.opponents[id_] = Target(opp, self.p)

    def can_bust(self) -> Optional[int]:
        """Return the id of a ghost if it can be busted, else None."""
        for id_, target in self.targets.items():
            if R_BUST_MIN <= target.d <= R_BUST_MAX:
                return id_
        return None

    def can_stun(self) -> Optional[int]:
        """Return the id of an opponent if it can be stun, else None."""
        for id_, opp in self.opponents.items():
            if opp.d <= R_BUST_MAX and opp.id not in stun_opponents:
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

    def get_closest_target(self) -> Optional[Target]:
        """Get the closest target if any."""
        if self.targets:
            return sorted(list(self.targets.values()), key=lambda t: t.d)[0]
        return None

    def explore(self, num) -> Point:
        """Return a Point allowing to explore the map if no ghost is visible."""
        debug(f"{num} explores")

        if not self.is_exploring:
            self.is_exploring = True
            point_index.setdefault(self.id, 0)

        next_point = PATH[num][point_index[self.id]]
        distance = self.dist_to(next_point)
        debug(
            f"next point: {next_point} ({point_index[self.id]})"
            f" from {self.p} ({distance})"
        )
        if distance < 2:
            point_index[self.id] += 1
            if point_index[self.id] == len(PATH[num]):
                point_index[self.id] = 0
            next_point = PATH[num][point_index[self.id]]
            debug(f"new point {next_point}")
        return next_point


ghost_registry: Dict[int, Ghost] = {}
last_stun: Dict[int, int] = {0:-100, 1:-100, 2:-100, 3:-100, 4:-100}

# game loop
if __name__ == '__main__':
    nb = 0
    while True:
        nb += 1


        # Load the visible entities
        entities = int(input())  # the number of busters and ghosts visible to you
        visible_ghosts = {}
        opponents = {}
        busters = []
        for i in range(entities):
            # entity_id: buster id or ghost id
            # y: position of this buster / ghost
            # entity: the team id if it is a buster, -1 if it is a ghost.
            # state: For busters: 0=idle, 1=carrying a ghost.
            # value: For busters: Ghost id being carried.
            #        For ghosts: number of busters attempting to trap this ghost.
            entity_id, x, y, entity, state, value = [int(j) for j in input().split()]
            if entity == MY_ID:
                busters.append(Buster(entity_id, x, y, state, value))
            elif entity == -1:
                visible_ghosts[entity_id] = Ghost(entity_id, x, y, value)
            else:
                opponents[entity_id] = Buster(entity_id, x, y, state, value)

        # Clean the ghost registry from ghost at sight distance
        for buster in busters:
            for ghost in list(ghost_registry.values()):
                if buster.dist_to(ghost.p) <= R_FOG:
                    del ghost_registry[ghost.id]

        # Merge the visible ghosts into the registry
        invisible_ghosts = deepcopy(ghost_registry)
        ghost_registry.update(visible_ghosts)

        for i, buster in enumerate(busters):
            if buster.loaded:
                if buster.dist_to(HOME) <= R_RELEASE:
                    print("RELEASE")
                    continue
                print(f"MOVE {HOME.x} {HOME.y}")
                continue

            if not NB_GHOSTS:
                print(f"MOVE {HOME.x} {HOME.y}")

            # Try to stun
            if nb - last_stun[i] > COOLDOWN:
                buster.compute_dist_to_opponents(opponents)
                target_id = buster.can_stun()
                if target_id is not None:
                    print(f"STUN {target_id}")
                    last_stun[i] = nb
                    stun_opponents[target_id] = nb
                    continue

            # No visible nor invisible ghosts, so exploring
            if not visible_ghosts and not invisible_ghosts:
                p = buster.explore(i)
                print(f"MOVE {p.x} {p.y}")
                continue

            # No visible but invisible ghosts
            buster.is_exploring = False
            if not visible_ghosts:
                buster.compute_dist_to_ghost(invisible_ghosts)
                target = buster.get_closest_target()
                del invisible_ghosts[target.id]
                debug(f"{i} targets INvisible {target.id}")
                print(f"MOVE {target.p.x} {target.p.y}")
                continue

            # Visible ghosts
            buster.compute_dist_to_ghost(visible_ghosts)
            for id_, t in buster.targets.items():
                debug(f"{id_} -> {t.d}")

            target_id = buster.can_bust()
            if target_id is not None:
                print(f"BUST {target_id}")
                del visible_ghosts[target_id]
                del ghost_registry[target_id]
                continue

            target_id = buster.has_too_close_target()
            if target_id is not None:
                p = buster.compute_bust_point(target_id)
                print(f"MOVE {p.x} {p.y}")
                del visible_ghosts[target_id]
                continue

            target = buster.get_closest_target()
            if target is not None:
                del visible_ghosts[target.id]
                debug(f"{i} targets visible {target.id}")
                print(f"MOVE {target.p.x} {target.p.y}")
                continue

            debug(f"ERROR: VISIBLE GHOST BUT NO TARGET FOR: {buster}")
            print(f"MOVE {X_MAX/2} {Y_MAX/2}")
