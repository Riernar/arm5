import datetime
from typing import TypeAlias

Value: TypeAlias = None | bool | int | float | str | datetime.datetime
Record: TypeAlias = dict[str, Value]
