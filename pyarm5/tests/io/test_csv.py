import pathlib
import tempfile

from arm5.io import csv

HERE = pathlib.Path(__file__).parent


def test_read():
    expected = [
        {"col1": "a", "col2": "b", "col3": "c"},
        {},
        {"col1": "d", "col2": "e"},
    ]
    HERE = pathlib.Path(__file__).parent
    assert expected == csv.read(HERE / "data/example.csv")


def test_write():
    with tempfile.TemporaryDirectory() as tdir:
        fpath = pathlib.Path(tdir) / "example.csv"
        expected = [
            {"col1": "a", "col2": "b", "col3": "c"},
            {},
            {"col1": "d", "col2": "e"},
        ]
        csv.write(fpath, expected)
        expected == csv.read(fpath)
