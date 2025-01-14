#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# ]
# ///

import argparse
import collections
import collections.abc
import contextlib
import csv
import dataclasses
import enum
import functools
import json
import logging
import operator
import pathlib
import re
import sys
import typing
from typing import Any, Final, Iterable, Iterator, Literal, Mapping, NoReturn, Sequence, TextIO

LOGGER = logging.getLogger(__name__)


class UserError(Exception):
    """Known error that should be reported to the user"""


@functools.total_ordering
class Season(enum.Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

    @functools.cached_property
    def _index(self) -> int:
        return next(index for index, member in enumerate(Season) if member is self)

    def __lt__(self, other: "Season") -> bool:
        if not isinstance(other, Season):
            return NotImplemented
        return self._index < other._index

    @classmethod
    def range(cls, start: "Season|None" = None, end: "Season | None" = None) -> "Iterable[Season]":
        start = start or Season.SPRING
        end = end or Season.WINTER
        yield from (season for season in cls if start <= season <= end)


class Inventory(Mapping[str, int]):
    """Immutable mapping from elements to counts, supporting addition and subtraction"""

    def __init__(self, obj: Mapping[str, int] | Iterable[str] | None = None, /, **kwargs: int):
        self._data = dict()
        # Copied from collections.Counter.update() to have the same semantics
        if obj is not None:
            if isinstance(obj, collections.abc.Mapping):
                self._data.update(obj)
            else:
                for elem in obj:
                    self._data[elem] = self._data.get(elem, 0) + 1
        if kwargs:
            for elem, count in kwargs.items():
                self._data[elem] = self._data.get(elem, 0) + count
        for k in list(self._data):  # we might del inside the loop, need to iterate on a copy
            if self._data[k] == 0:
                del self._data[k]

    # collections.abc.Mapping minimal implementation. The inheritance will fill-in the Mapping protocol

    def __getitem__(self, key: str) -> int:
        return self._data.get(key, 0)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    # Implement get() directly to allow for the default value as __getitem__ returns 0

    @typing.overload
    def get(self, key: str) -> int: ...
    @typing.overload
    def get[T](self, key: str, /, default: T) -> int | T: ...

    def get(self, key: str, /, default=0):  # type: ignore
        return self._data.get(key, default)

    # Implement addition and subtraction

    @classmethod
    def _from_dict(cls, *args, **kwargs) -> "Inventory":
        result = cls()
        result._data = dict(*args, **kwargs)
        return result

    def __pos__(self) -> "Inventory":
        return Inventory._from_dict((k, +v) for k, v in self.items() if v != 0)

    def __neg__(self) -> "Inventory":
        return Inventory._from_dict((k, -v) for k, v in self.items() if v != 0)

    def __add__(self, other: "Inventory") -> "Inventory":
        if not isinstance(other, Inventory):
            return NotImplemented
        return Inventory._from_dict(
            (k, new_count) for k in self.keys() | other.keys() if (new_count := self[k] + other[k]) != 0
        )

    def __sub__(self, other: "Inventory") -> "Inventory":
        if not isinstance(other, Inventory):
            return NotImplemented
        return Inventory._from_dict(
            (k, new_count) for k in self.keys() | other.keys() if (new_count := self[k] - other[k]) != 0
        )


@dataclasses.dataclass(frozen=True)
class VisRecord:
    description: str
    vis: Mapping[str, int]
    year: int
    season: Season
    magus: str | None = None
    period: int | None = None
    end_year: int | None = None

    def __post_init__(self) -> None:
        """Simple-minded data validation"""
        object.__setattr__(self, "year", int(self.year))
        object.__setattr__(self, "season", Season(self.season))
        object.__setattr__(self, "magus", self.magus if self.magus else None)
        object.__setattr__(self, "period", int(self.period) if self.period else None)
        object.__setattr__(self, "end_year", int(self.end_year) if self.end_year else None)

    def is_active_on(self, year: int, season: Season) -> bool:
        if self.period is None:
            return self.year == year and self.season == season
        else:
            if self.season != season:
                return False
            if self.year > year:
                return False
            if self.end_year is not None and year > self.end_year:
                return False
            return ((year - self.year) % self.period) == 0


@dataclasses.dataclass(frozen=True)
class VisLedger:
    """Description of vis stocks & consumption in time, and by magi."""

    start: tuple[int, Season]
    start_stock: Inventory
    end: tuple[int, Season]
    # year -> season -> record[]
    ledger: Mapping[int, Mapping[Season, tuple[VisRecord, ...]]]
    # year -> season -> stock
    stocks: Mapping[int, Mapping[Season, Inventory]]
    magi_uses: Mapping[str, Inventory]

    @classmethod
    def from_records(
        cls,
        records: Iterable[VisRecord],
        *,
        start_stock: Mapping[str, int] | None = None,
        end_year: int | None = None,
        end_season: Season | None = None,
    ) -> "VisLedger":
        """Compute a full ledger from initial stocks and vis records"""
        records = list(records)

        start_year = min(record.year for record in records)
        start_season = min((record.season for record in records if record.year == start_year), default=Season.SPRING)

        end_year = end_year or max(record.year for record in records)
        end_season = end_season or max(
            (record.season for record in records if record.year == end_year), default=Season.WINTER
        )

        stock = start_stock = Inventory(start_stock or {})

        ledger: dict[int, Mapping[Season, tuple[VisRecord, ...]]] = {}
        stocks: dict[int, Mapping[Season, Inventory]] = {}

        for year in range(start_year, end_year + 1):
            year_ledger: dict[Season, tuple[VisRecord, ...]] = {}
            year_stocks: dict[Season, Inventory] = {}

            first_season = start_season if year == start_year else Season.SPRING
            last_season = end_season if year == end_year else Season.WINTER
            for season in Season.range(first_season, last_season):
                year_ledger[season] = tuple(record for record in records if record.is_active_on(year, season))
                stock_update = sum((Inventory(record.vis) for record in year_ledger[season]), start=Inventory())
                stock = stock + stock_update
                year_stocks[season] = stock

            ledger[year] = year_ledger
            stocks[year] = year_stocks

        magi = {record.magus for record in records if record.magus}
        magi_uses = {
            magus: sum(
                (-Inventory(record.vis) for record in records if record.magus == magus),
                start=Inventory(),
            )
            for magus in magi
        }

        return VisLedger(
            start=(start_year, start_season),
            start_stock=start_stock,
            end=(end_year, end_season),
            ledger=ledger,
            stocks=stocks,
            magi_uses=magi_uses,
        )


def cli() -> None:
    parser = _get_arg_parser()
    args = parser.parse_args()
    _setup_logging()
    try:
        with _open_input_output(args) as (inputs, outputs):
            start_stock, records = _READERS[inputs.format](inputs.file)
            ledger = VisLedger.from_records(
                records, start_stock=start_stock, end_year=args.end_year, end_season=args.end_season
            )
            _WRITERS[outputs.format](ledger, outputs.file)

    except UserError as err:
        _error(err)


def _get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI program to manage vis stocks")

    inputs = parser.add_argument_group(title="inputs", description="inputs")
    inputs.add_argument(
        "-i",
        "--input",
        default=sys.stdin,
        type=pathlib.Path,
        help="Input file to read input data from. Defaults to STDIN.",
    )
    inputs.add_argument(
        "--input-format",
        default=None,
        choices=["json", "csv"],
        help=(
            "Input format. Defaults to 'json' if reading from STDIN,"
            " or detected from input file extension if reading from a file."
        ),
    )

    outputs = parser.add_argument_group(title="outputs", description="outputs")
    outputs.add_argument(
        "-o", "--output", default=sys.stdout, type=pathlib.Path, help="Output path. Defaults to STDOUT"
    )
    outputs.add_argument(
        "--output-format",
        default=None,
        choices=["json", "md"],
        help=(
            "Input format. Defaults to 'json' if writing to STDOUT,"
            " or inferred from output file extension if writing to file."
        ),
    )
    parser.add_argument("--end-year", type=int, default=None, help="Last year to include in the ledger")
    parser.add_argument("--end-season", type=Season, default=None, help="Last season to include in the ledger")
    return parser


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclasses.dataclass(frozen=True)
class _FormatIO:
    file: TextIO
    format: str


@contextlib.contextmanager
def _open_input_output(args: argparse.Namespace) -> Iterator[tuple[_FormatIO, _FormatIO]]:
    input_format = args.input_format or _infer_format(args.input, default="json", supported={"json", "csv"})
    output_format = args.output_format or _infer_format(args.output, default="json", supported={"json", "md"})

    with (
        _open_file_or_stream(args.input, mode="r") as infile,
        _open_file_or_stream(args.output, mode="w") as outfile,
    ):
        # Notify the user to avoid confusion
        if infile is sys.stdin:
            LOGGER.info("Reading from STDIN")
        yield _FormatIO(infile, input_format), _FormatIO(outfile, output_format)


def _infer_format(arg: TextIO | pathlib.Path, *, default: str, supported: set[str]) -> str:
    if isinstance(arg, pathlib.Path):
        # The slice [1:] removes the leading dot that is included in the suffix by python
        extension = arg.suffixes[-1][1:]
        if extension not in supported:
            raise UserError(f"Unsupported format '{extension}' in {arg}: supported formats are {sorted(supported)}")
        return extension

    return default


def _error(err: Exception) -> NoReturn:
    if sys.stdout.isatty():
        template = "\033[1;91m{}\033[0m"
    else:
        template = "{}"
    print(template.format(str(err)), file=sys.stderr, flush=True)
    sys.exit(1)


@contextlib.contextmanager
def _open_file_or_stream(arg: TextIO | pathlib.Path, mode: Literal["r", "w"]) -> Iterator[TextIO]:
    if isinstance(arg, pathlib.Path):
        try:
            with open(arg, mode, encoding="utf-8") as file:
                yield file
        except (FileNotFoundError, NotADirectoryError) as err:
            raise UserError(f"{type(err).__name__}: {err}")

    else:
        yield arg


def _read_json(stream: TextIO) -> tuple[Mapping[str, int], tuple["VisRecord", ...]]:
    raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level JSON Object, got {type(raw)}")
    start_stock = {vis: int(quantity) for vis, quantity in raw.get("start_stock", {}).items()}
    records = tuple(VisRecord(**data) for data in raw.get("records", []))
    return start_stock, records


def _read_csv(stream: TextIO) -> tuple[Mapping[str, int], tuple["VisRecord", ...]]:
    reader = csv.DictReader(stream)

    # Build the fields we need to have in the CSV
    fields = {field.name for field in dataclasses.fields(VisRecord)} - {"vis"}
    key_fields = sorted(fields)
    fields |= {"art", "amount"}

    headers = set(reader.fieldnames)  # type: ignore
    if unknown_headers := headers - fields:
        raise UserError(f"Invalid CSV: unknown headers {sorted(unknown_headers)}: supported headers: {sorted(fields)}")
    if missing_headers := fields - headers:
        raise UserError(f"Invalid CSV:missing headers {sorted(missing_headers)}")

    # When reading CSV, we need to agglomerate together lines that express several vis types for a single action
    start_stock = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    get_key = operator.itemgetter(*key_fields)
    for row in reader:
        art = row["art"]
        amount = int(row["amount"])
        if row["year"] in {None, ""}:
            start_stock[art] += amount
        else:
            key = get_key(row)
            groups[key][art] += amount
    records = tuple(VisRecord(vis=Inventory(vis), **dict(zip(key_fields, key))) for key, vis in groups.items())
    return Inventory(start_stock), records


_READERS: Final = {
    "json": _read_json,
    "csv": _read_csv,
}


def _write_json(ledger: VisLedger, stream: TextIO) -> None:
    json.dump(_to_json(ledger), stream, indent=2)


def _to_json(obj: Any, path=tuple()) -> Any:
    if isinstance(obj, (type(None), bool, int, float, str)):
        return obj
    elif isinstance(obj, (Sequence, set, frozenset)):
        return [_to_json(element, (*path, index)) for index, element in enumerate(obj)]
    elif isinstance(obj, Mapping):
        return {_to_str(k): _to_json(v, (*path, k)) for k, v in obj.items()}
    elif isinstance(obj, enum.Enum):
        return _to_json(obj.value)
    elif dataclasses.is_dataclass(obj):
        return {
            field.name: _to_json(value)
            for field in dataclasses.fields(obj)
            if (value := getattr(obj, field.name, dataclasses.MISSING)) is not dataclasses.MISSING
        }
    raise TypeError(f"Unsupported JSON conversion for {type(obj).__name__} object")


def _to_str(obj) -> str:
    if isinstance(obj, enum.Enum):
        return _to_str(obj.value)
    return str(obj)


def _write_markdown(ledger: VisLedger, stream: TextIO) -> None:
    _writelines(stream, _md_header())
    _writelines(stream, _md_magi_uses(ledger))
    _writelines(stream, _md_yearly_stocks(ledger))
    _writelines(stream, _md_records(ledger))


def _writelines(stream: TextIO, lines: Iterable[str], end: str = "\n") -> None:
    """Wrapper around TextIO.writelines() that handles line ending"""
    stream.writelines(f"{line}{end}" for line in lines)


def _md_header() -> Iterable[str]:
    return [
        "# Vis",
        "",
        "**WARNING**: This file is automatically generated, do _not_ edit directly",
    ]


def _md_magi_uses(ledger: VisLedger) -> Iterable[str]:
    yield ""
    yield "## Magi consumption"
    yield ""
    vis_types = set().union(*(uses.keys() for uses in ledger.magi_uses.values()))
    vis_types = _sort_vis_names(vis_types, all_vis=True)
    magi = [*sorted(ledger.magi_uses)]
    columns = {"": [magus.title() for magus in magi]}

    for vis in vis_types:
        columns[vis] = [str(ledger.magi_uses[magus].get(vis, "")) for magus in magi]
    yield from _md_table(columns)


def _md_yearly_stocks(ledger: VisLedger) -> Iterable[str]:
    yield ""
    yield "## Yearly stocks"
    yield ""

    vis_types_set = set(ledger.start_stock).union(
        *(inventory.keys() for season_stocks in ledger.stocks.values() for inventory in season_stocks.values())
    )
    vis_types = _sort_vis_names(vis_types_set, all_vis=True)
    years = sorted(ledger.stocks, reverse=True)
    columns = {"year": list(map(str, years))}
    for vis in vis_types:
        column: list[str] = []
        for year in years:
            year_stocks = ledger.stocks[year]
            last_season = max(year_stocks)
            column.append(str(year_stocks[last_season].get(vis, "")))
        columns[vis] = column

    if ledger.start_stock:
        columns["year"].append("start")
        for vis in vis_types:
            columns[vis].append(str(ledger.start_stock.get(vis, "")))

    yield from _md_table(columns)


def _md_records(ledger: VisLedger) -> Iterable[str]:
    yield ""
    yield "## Movements"
    yield ""
    for year in sorted(ledger.ledger):
        yield f"- **{year}**"

        year_records = ledger.ledger[year]
        for season in Season:
            if season in year_records:
                yield f"  - **{season.value.title()}**"

                for record in year_records[season]:
                    # NOTE: In python 3.7+ python dictionaries preserve order and this is part of the API
                    vis = {vis: record.vis[vis] for vis in _sort_vis_names(record.vis)}
                    vis_pos = ", ".join(f"{amount:+} {vis.title()}" for vis, amount in vis.items() if amount > 0)
                    vis_neg = ", ".join(f"{amount:+} {vis.title()}" for vis, amount in vis.items() if amount < 0)
                    vis_diff = ", ".join(filter(None, [vis_pos, vis_neg]))  # add a separator if necessary
                    if record.magus:
                        yield f"    - _{record.description}_ ({record.magus.title()}): {vis_diff}"
                    else:
                        yield f"    - _{record.description}_: {vis_diff}"


_split_name = re.compile(r"[.,;:/|+&-]").split


def _sort_vis_names(names: Iterable[str], *, all_vis: bool = False) -> list[str]:
    arts = ["cr", "in", "mu", "pe", "re", "an", "aq", "au", "co", "he", "ig", "im", "me", "te", "vi"]
    indices = {art: index for index, art in enumerate(arts)}

    def key(name: str) -> tuple[int, ...]:
        parts = [part.strip().lower() for part in _split_name(name)]
        return (len(parts), *tuple(indices.get(part.lower()[:2], len(part)) for part in parts))

    if all_vis:
        names = [*arts, *names]

    return sorted(names, key=key)


def _md_table(columns: dict[str, Any]) -> Iterable[str]:
    headers = tuple(columns)
    widths = [max(3, len(col), max(map(len, columns[col]))) for col in headers]
    # first column is right aligned, then everything is centered
    alignments = ">" + "^" * (len(headers) - 1)

    def _row(cells: Iterable[str]) -> str:
        cells = [f" {row: {align}{width}} " for row, align, width in zip(cells, alignments, widths, strict=True)]
        return "|" + "|".join(cells) + "|"

    yield _row(header.title() for header in headers)
    yield _md_alignment_row(alignments, widths)
    rows: Iterable[tuple[str, ...]] = zip(*(columns[col] for col in headers))
    yield from (_row(row) for row in rows)


def _md_alignment_row(alignment: str, widths: Iterable[int]) -> str:
    fstring_to_md_align = {">": _md_table_align_right, "^": _md_table_align_center}
    cells = (fstring_to_md_align[align](width) for align, width in zip(alignment, widths, strict=True))
    return "|" + "|".join(cells) + "|"


def _md_table_align_right(width: int) -> str:
    return " " + "-" * (width - 1) + ": "


def _md_table_align_center(width: int) -> str:
    return " :" + ("-" * max(0, width - 2)) + ": "


_WRITERS: Final = {
    "json": _write_json,
    "md": _write_markdown,
}

if __name__ == "__main__":
    cli()
