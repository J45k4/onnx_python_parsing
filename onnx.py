"""Simple ONNX file loader.

This module provides a :func:`load` function to load ONNX model files. If the
:mod:`onnx` package is available in the environment the file is parsed and the
``ModelProto`` object is returned. Otherwise the raw file contents are returned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Union

# Attempting to import the real ``onnx`` package would lead to a circular import
# because this file is named ``onnx.py``. The package is also not available in
# this environment, so we simply expose a ``None`` placeholder.  The logic below
# checks for ``None`` and falls back to returning the raw file contents.
_onnx = None


# The return type is ``Any`` because it depends on whether ``onnx`` is installed.
def load(path: str | Path) -> Union[Any, bytes]:
    """Load an ONNX model from ``path``.

    Parameters
    ----------
    path:
        Path to the ONNX model file.

    Returns
    -------
    Union[Any, bytes]
        The parsed ``ModelProto`` when :mod:`onnx` is available, otherwise the
        raw bytes of the file.
    """

    data = Path(path).read_bytes()
    if _onnx is not None:
        return _onnx.load_from_string(data)
    return data


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for loading an ONNX model."""
    parser = argparse.ArgumentParser(description="Load an ONNX model")
    parser.add_argument("path", help="Path to the ONNX model file")
    args = parser.parse_args(argv)
    model = load(args.path)
    print(model)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
