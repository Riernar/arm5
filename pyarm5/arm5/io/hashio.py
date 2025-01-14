"""
Module with utilities to generate file hashes on write, and validate such hashes on read

This module provide a wrapper around `builtins.open` that:
- validate the hash of the opened file, if any
- hash the file and save the hash after closing

This allows detecting that a generated file has been manually modified since it was generated and
avoid silently overwriting such modifications.
"""
import builtins
import dataclasses
import functools
import hashlib
import inspect
import io
import pathlib
import typing
import warnings
from typing import Callable, Literal, TypeVar


@dataclasses.dataclass(frozen=True)
class _HashFileWrapper:
    """
    Wrapper around a file-object that generate a filehash after closing the file
    """

    file: io.IOBase
    path: pathlib.Path | None = None
    hash_path: pathlib.Path | None = None
    hash_name: str = "sha256"

    def __enter__(self):
        return self.file

    def __exit__(self, *args, **kwargs):
        ret = self.file.__exit__(*args, **kwargs)
        if self.path is not None and self.hash_path is not None:
            self.hash_path.write_bytes(path_digest(self.path, self.hash_name))
        return ret


# We need to capture the type of open() so that type-checker thing the wrapper is used like open()


OpenFuncT = TypeVar("OpenFuncT", bound=Callable)


def wrap_open(
    open: OpenFuncT,
    *,
    hash_on_write: bool = True,
    on_error: Literal["raise", "warn", "ignore"] = "raise",
    hash_name: str = "sha256",
    # lie on the return type, this is the only way to capture open() very complex signature
) -> OpenFuncT:
    """
    Builds a wrapper around `builtins.open` that validate and write file hashes

    When a file is opened with the returned wrapper, it looks for a corresponding hash file.
    The hash file has the same name and path, and an additional suffix that is the hashing
    algorithm used (by default "sha256"). Before opening the file, the current hash of the file is
    compared against the stored hash to verify integrity.

    If a file is opened in "write" or "append" mode, the hash file is then deleted (this avoids
    an incorrect hash file lingering in the event of a crash). The hash file is re-created with
    the new hash after the file is closed.

    As we need to wrap around the call to `open()`, both before the file has been opened and after
    it has been closed, the returned wrapper *must* be used like a context manager, the same way
    that `open` is used.
    """

    if open is not builtins.open:
        raise TypeError(f"Can only wrap builtins.open ({builtins.open}), got {open}")

    @functools.wraps(open)
    def hash_open(*args, **kwargs):
        """
        Wrapper around `builtins.open` that validate and write file hashes.

        When a file is opened with this wrapper, it looks for a corresponding hash file.
        The hash file has the same name and path, and an additional suffix that is the hashing
        algorithm used (by default "sha256"). Before opening the file, the current hash of the file
        is compared against the stored hash to verify integrity.

        If a file is opened in "write" or "append" mode, the hash file is then deleted (this avoids
        an incorrect hash file lingering in the event of a crash). The hash file is re-created with
        the new hash after the file is closed.

        As we need to wrap around the call to `open`, both before the file has been opened and
        after it has been closed, this wrapper *must* be used like a context manager, the same way
        that `open` is used.

        All arguments are forwarded to `builtins.open`.

        To control the exact behavior of this wrapper, see `arm5.io.hashio.wrap_open` and it's
        arguments.
        """
        # verify early-on that the hash name is valid
        hashlib.new(hash_name, usedforsecurity=False)

        # Extract the file that is being opened
        params = inspect.signature(open).bind(*args, **kwargs)
        params.apply_defaults()
        path = pathlib.Path(params.arguments["file"])
        mode = params.arguments["mode"]

        # Compute the associated hash file
        hash_suffix = f".{path.suffix}.{hash_name}" if path.suffix else f".{hash_name}"
        hash_path = path.with_suffix(hash_suffix)

        # Validate the file is applicable
        if on_error != "ignore" and path.is_file() and hash_path.is_file():
            validate_path(path, hash_name, hash_path.read_bytes(), on_error=on_error)

        # Remove the saved hash file during edition to avoid wrong hashfile lingering being
        if any(m in mode for m in "wa") and hash_path.is_file():
            hash_path.unlink()

        # Open the file and schedule hashing on close if applicable
        file = builtins.open(*args, **kwargs)
        if any(m in mode for m in "wa") and hash_on_write:
            return _HashFileWrapper(file, path, hash_path, hash_name)
        return _HashFileWrapper(file)

    return hash_open  # type: ignore


def path_digest(path: pathlib.Path, digest: str | Callable[[], "hashlib._Hash"], /) -> bytes:
    """Wrapper for `hashlib.file_digest` that open a path and forward the file to hashlib"""
    with builtins.open(path, "rb") as fp:
        return hashlib.file_digest(fp, digest).digest()


def validate_path(
    path: pathlib.Path,
    digest: str | Callable[[], "hashlib._Hash"],
    expected: bytes,
    *,
    on_error: Literal["raise", "warn"] = "raise",
) -> None:
    """Verify that a file has the expected hash"""
    actual = path_digest(path, digest)
    if expected != actual:
        msg = (
            f"{path} has been modified since it was last generated: "
            f" expected {expected.hex()} digest, got {actual.hex()}"
        )
        if on_error == "raise":
            raise OSError(msg)
        elif on_error == "warn":
            warnings.warn(msg)
        else:
            typing.assert_never(on_error)


# Save a wrapped version of open()
open = wrap_open(builtins.open)
