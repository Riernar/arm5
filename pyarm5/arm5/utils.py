"""
Various utilities
"""

from typing import Iterable


def quote(obj: object | Iterable[str]) -> str:
    """Stringify and quote object"""
    # Str are iterable of their individual letters, but that's not what we want
    if isinstance(obj, str):
        return f'"{obj}"'
    iterable: Iterable
    try:
        # try to iterate on the object
        iterable = iter(obj)  # type: ignore
    except (TypeError, ValueError):
        # object isn't iterable, wrap it so it is stringified on its own
        iterable = [obj]
    # stringify, quote and join with commas
    return ", ".join(f"'{s}'" for s in sorted(map(str, iterable)))
