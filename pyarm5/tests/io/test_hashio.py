import pathlib
import tempfile

import pytest

from arm5.io import hashio


def test_path_digest():
    with tempfile.TemporaryDirectory() as tdir:
        path = pathlib.Path(tdir) / "testfile"
        path.write_bytes(b"123456")
        expected = (
            b"\x8d\x96\x9e\xefn\xca\xd3\xc2\x9a:b\x92\x80\xe6\x86\xcf\x0c?]Z"
            b"\x86\xaf\xf3\xca\x12\x02\x0c\x92:\xdcl\x92"
        )
        assert hashio.path_digest(path, "sha256") == expected


def test_validate():
    with tempfile.TemporaryDirectory() as tdir:
        path = pathlib.Path(tdir) / "testfile"
        path.write_bytes(b"123456")
        expected = (
            b"\x8d\x96\x9e\xefn\xca\xd3\xc2\x9a:b\x92\x80\xe6\x86\xcf\x0c?]Z"
            b"\x86\xaf\xf3\xca\x12\x02\x0c\x92:\xdcl\x92"
        )
        hashio.validate_path(path, "sha256", expected)
        with pytest.raises(OSError):
            hashio.validate_path(path, "sha256", b"12345")


def test_open():
    with tempfile.TemporaryDirectory() as tdir:
        path = pathlib.Path(tdir) / "testfile"
        hash_path = path.with_suffix(".sha256")

        path.write_bytes(b"123456")
        expected = (
            b"\x8d\x96\x9e\xefn\xca\xd3\xc2\x9a:b\x92\x80\xe6\x86\xcf\x0c?]Z"
            b"\x86\xaf\xf3\xca\x12\x02\x0c\x92:\xdcl\x92"
        )

        # Test validation doesn't raise an error when file is valid
        hash_path.write_bytes(expected)
        with hashio.open(path):
            pass

        # test validation does raise when file is invalid
        hash_path.write_bytes(b"12345")
        with pytest.raises(OSError):
            with hashio.open(path):
                pass

        # test writing corrects the hash
        hash_path.write_bytes(expected)
        with hashio.open(path, "w") as file:
            file.write("changed content")
        new_expected = (
            b"\xb9-\x13\xbb\xe0-\xb7\xcav\x86\xa8\xe7\xb8T\xdeI\xc7EYH\xc0\\\xf9\x1a"
            b"G\x04Bx9^!."
        )
        assert hash_path.read_bytes() == new_expected
