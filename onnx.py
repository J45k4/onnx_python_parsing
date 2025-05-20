"""Minimal ONNX parser without external dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint starting at ``pos`` in ``buf``."""
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7


def skip_unknown(wtype: int, buf: bytes, pos: int) -> int:
    """Skip over an unknown field given its wire type."""
    if wtype == 0:  # varint
        _, pos = read_varint(buf, pos)
    elif wtype == 1:  # 64-bit
        pos += 8
    elif wtype == 2:  # length-delimited
        length, pos = read_varint(buf, pos)
        pos += length
    elif wtype == 5:  # 32-bit
        pos += 4
    else:
        raise ValueError(f"unsupported wire type {wtype}")
    return pos


def parse_string(buf: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_varint(buf, pos)
    s = buf[pos : pos + length].decode("utf-8", "ignore")
    return s, pos + length


def parse_nodeproto(buf: bytes) -> Dict[str, Any]:
    node: Dict[str, Any] = {"input": [], "output": []}
    pos = 0
    while pos < len(buf):
        tag, pos = read_varint(buf, pos)
        field, wtype = tag >> 3, tag & 0x07
        if field in (1, 2) and wtype == 2:  # input/output
            val, pos = parse_string(buf, pos)
            key = "input" if field == 1 else "output"
            node[key].append(val)
        elif field == 3 and wtype == 2:  # op_type
            val, pos = parse_string(buf, pos)
            node["op_type"] = val
        elif field == 4 and wtype == 2:  # name
            val, pos = parse_string(buf, pos)
            node["name"] = val
        else:
            pos = skip_unknown(wtype, buf, pos)
    return node


def parse_graphproto(buf: bytes) -> Dict[str, Any]:
    graph: Dict[str, Any] = {"node": []}
    pos = 0
    while pos < len(buf):
        tag, pos = read_varint(buf, pos)
        field, wtype = tag >> 3, tag & 0x07
        if field == 1 and wtype == 2:  # node
            length, pos = read_varint(buf, pos)
            node_bytes = buf[pos : pos + length]
            pos += length
            graph["node"].append(parse_nodeproto(node_bytes))
        elif field == 4 and wtype == 2:  # name
            val, pos = parse_string(buf, pos)
            graph["name"] = val
        else:
            pos = skip_unknown(wtype, buf, pos)
    return graph


def parse_modelproto(buf: bytes, pos: int = 0) -> Dict[str, Any]:
    model: Dict[str, Any] = {}
    while pos < len(buf):
        tag, pos = read_varint(buf, pos)
        field, wtype = tag >> 3, tag & 0x07
        if field == 1 and wtype == 0:  # ir_version
            val, pos = read_varint(buf, pos)
            model["ir_version"] = val
        elif field == 7 and wtype == 2:  # graph
            length, pos = read_varint(buf, pos)
            graph_bytes = buf[pos : pos + length]
            pos += length
            model["graph"] = parse_graphproto(graph_bytes)
        else:
            pos = skip_unknown(wtype, buf, pos)
    return model


def load(path: str | Path) -> Dict[str, Any]:
    """Load ``path`` and parse it as a minimal ONNX model."""
    data = Path(path).read_bytes()
    return parse_modelproto(data)


def main(argv: List[str] | None = None) -> None:
    """Command-line interface for the ONNX parser."""
    parser = argparse.ArgumentParser(description="Parse an ONNX model")
    parser.add_argument("path", help="Path to the ONNX model file")
    args = parser.parse_args(argv)
    model = load(args.path)
    print(model)


if __name__ == "__main__":
    main()
