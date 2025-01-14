import enum
import functools
from typing import Iterable, Self, TypeAlias

import pydantic

from arm5 import bases

Year: TypeAlias = pydantic.PositiveInt


class Seasons(enum.Enum):
    """Enumeration for the Season system of Ars Magica 5th"""

    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn", ["fall"]
    WINTER = "winter"

    # Aliases (python side)
    FALL = AUTUMN

    # handle aliases string side

    # An attribute to store the aliases of a member, if any
    _aliases_: frozenset[str]

    # When creating a value an class-creation time, handle aliases (see AUTUMN above)
    def __new__(cls, value: str, aliases: Iterable[str] | None = None):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._aliases_ = frozenset(aliases) if aliases is not None else frozenset()
        return obj

    # When de-serializing, allow using the aliases of members
    @classmethod
    def _missing_(cls, value: object) -> Self:
        if isinstance(value, str):
            for member in cls:
                if value in member._aliases_:
                    return member
        return super()._missing_(value)

    # Seasons are ordered, support ordering based on declaration order

    @property
    def _enumeration_order(self) -> int:
        return list(type(self).__members__.values()).index(self)

    # Implementation based on https://docs.python.org/3/howto/enum.html#orderedenum

    def __ge__(self, other):
        if isinstance(other, type(self)):
            return self._enumeration_order >= other._enumeration_order

        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, type(self)):
            return self._enumeration_order > other._enumeration_order

        return NotImplemented

    def __le__(self, other):
        if isinstance(other, type(self)):
            return self._enumeration_order <= other._enumeration_order

        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, type(self)):
            return self._enumeration_order < other._enumeration_order

        return NotImplemented


@functools.total_ordering
class YearSeason(bases.FrozenModel):
    year: Year
    season: Seasons

    @staticmethod
    def iter_between(start: "YearSeason", end: "YearSeason") -> Iterable["YearSeason"]:
        if end < start:
            return

        for season in Seasons:
            if season >= start.season:
                yield YearSeason(year=start.year, season=season)
        for year in range(start.year + 1, end.year):
            for season in Seasons:
                yield YearSeason(year=year, season=season)
        for season in Seasons:
            if season <= end.season:
                yield YearSeason(year=end.year, season=season)

    @classmethod
    def start_of_year(cls, year: Year) -> Self:
        return cls(year=year, season=Seasons.SPRING)

    @classmethod
    def end_of_year(cls, year: Year) -> Self:
        return cls(year=year, season=Seasons.WINTER)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, YearSeason):
            return self.year < other.year or (
                self.year == other.year and self.season < other.season
            )
        return NotImplemented
