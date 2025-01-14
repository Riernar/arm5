import csv
import dataclasses
import pathlib
from typing import Any, Iterable, Iterator, Self

from arm5.typing import Record


class CSVError(ValueError):
    """Raised when error occure with CSV files"""


@dataclasses.dataclass
class DictReader:
    """Variant of the builtin `csv.DictReader` that doesn't include empty columns in rows"""

    reader: Any
    missing_value: Any = None
    line_num: int = dataclasses.field(default=0, init=False)
    _fieldnames: list[str] | None = dataclasses.field(default=None, init=False)

    @classmethod
    def from_file(
        cls,
        /,
        file: Iterator[str],
        missing_value: Any = None,
        dialect="excel",
        *args,
        **kwargs,
    ) -> Self:
        return cls(
            reader=csv.reader(file, dialect, *args, **kwargs),
            missing_value=missing_value,
        )

    def __iter__(self):
        return self

    @property
    def fieldnames(self) -> list[str]:
        if self._fieldnames is None:
            try:
                self._fieldnames = next(self.reader)
            except StopIteration:
                pass
        self.line_num = self.reader.line_num
        return self._fieldnames  # type: ignore

    @fieldnames.setter
    def fieldnames(self, value):
        self._fieldnames = value

    def __next__(self):
        if self.line_num == 0:
            # Used only for its side effect.
            self.fieldnames
        row = next(self.reader)
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == []:
            row = next(self.reader)
        # convert empty strings to missing value
        row = [s if len(s) else self.missing_value for s in row]
        # Remove empty values from the dict
        d = dict((f, v) for f, v in zip(self.fieldnames, row) if v is not None)
        return d


def read(filepath: pathlib.Path) -> list[Record]:
    """Read a CSV file, returning a list of records"""
    with filepath.open(mode="rt", encoding="utf-8", newline="") as file:
        reader = DictReader.from_file(file)
        return list(reader)


def write(filepath: pathlib.Path, records: Iterable[Record]) -> pathlib.Path:
    """Write records to a CSV file"""
    records = list(records)
    columns = set().union(*(record.keys() for record in records))

    with filepath.open(mode="wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(records)
    return filepath
