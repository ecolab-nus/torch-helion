"""OpGraph: the loom-style operator graph used by every torch-helion stage.

An :class:`OpGraph` is a single-output-per-node DAG. Nodes are stored in
insertion order, which every producer keeps topological. Compared with the
FX graph it is built from, the OpGraph

* carries a concrete :class:`TensorType` (shape + dtype) on every node,
* keeps parameter *values* next to the graph (``graph.params``) so that the
  optimizer can fold weight transforms offline,
* uses a small, explicit op vocabulary (see :mod:`torch_helion.capture.ops`)
  instead of ATen overloads, and
* serialises to JSON so intermediate results can be stored under
  ``results/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "bool": 1}


@dataclass(frozen=True)
class TensorType:
    """Static shape and dtype of a graph value."""

    shape: tuple[int, ...]
    dtype: str = "float16"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES.get(self.dtype, 4)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def with_shape(self, shape: Iterable[int]) -> "TensorType":
        return TensorType(tuple(int(s) for s in shape), self.dtype)

    @staticmethod
    def of(t: torch.Tensor) -> "TensorType":
        return TensorType(tuple(int(s) for s in t.shape), str(t.dtype).replace("torch.", ""))

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(map(str, self.shape))}]"


NODE_KINDS = ("input", "param", "const", "compute", "layout")


@dataclass
class OpNode:
    """One operation producing one tensor."""

    name: str
    op: str
    inputs: list[str]
    type: TensorType
    attrs: dict[str, Any] = field(default_factory=dict)
    kind: str = "compute"
    origin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "shape": list(self.type.shape),
            "dtype": self.type.dtype,
            "attrs": _jsonable(self.attrs),
            "origin": self.origin,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OpNode":
        return OpNode(
            name=d["name"],
            op=d["op"],
            inputs=list(d["inputs"]),
            type=TensorType(tuple(d["shape"]), d["dtype"]),
            attrs=dict(d.get("attrs", {})),
            kind=d.get("kind", "compute"),
            origin=d.get("origin"),
        )

    def __str__(self) -> str:
        attrs = f" {self.attrs}" if self.attrs else ""
        return f"{self.name}: {self.type} = {self.op}({', '.join(self.inputs)}){attrs}"


class OpGraph:
    """Ordered DAG of :class:`OpNode`."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, OpNode] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.params: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, Any] = {}
        self._counter = 0

    # ------------------------------------------------------------------ build
    def fresh_name(self, prefix: str) -> str:
        while True:
            self._counter += 1
            name = f"{prefix}_{self._counter}"
            if name not in self.nodes:
                return name

    def add(
        self,
        op: str,
        inputs: Iterable[str],
        type: TensorType,
        attrs: dict[str, Any] | None = None,
        kind: str = "compute",
        name: str | None = None,
        origin: str | None = None,
        value: torch.Tensor | None = None,
    ) -> OpNode:
        """Append a node. ``value`` stores a concrete tensor for params/consts."""
        inputs = list(inputs)
        for src in inputs:
            if src not in self.nodes:
                raise KeyError(f"{op}: unknown input '{src}'")
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {kind!r}")
        name = name or self.fresh_name(op)
        if name in self.nodes:
            raise KeyError(f"duplicate node name {name!r}")
        node = OpNode(name, op, inputs, type, dict(attrs or {}), kind, origin)
        self.nodes[name] = node
        if kind == "input":
            self.inputs.append(name)
        if value is not None:
            self.params[name] = value
        return node

    def add_const(self, value: torch.Tensor, name: str | None = None, origin: str | None = None) -> OpNode:
        return self.add("const", [], TensorType.of(value), kind="const", name=name, origin=origin, value=value)

    # ---------------------------------------------------------------- queries
    def __getitem__(self, name: str) -> OpNode:
        return self.nodes[name]

    def __contains__(self, name: str) -> bool:
        return name in self.nodes

    def __iter__(self) -> Iterator[OpNode]:
        return iter(list(self.nodes.values()))

    def __len__(self) -> int:
        return len(self.nodes)

    def users(self, name: str) -> list[OpNode]:
        return [n for n in self.nodes.values() if name in n.inputs]

    def num_uses(self, name: str) -> int:
        """Number of operand slots (plus graph outputs) reading ``name``."""
        uses = sum(n.inputs.count(name) for n in self.nodes.values())
        return uses + self.outputs.count(name)

    def producers(self, node: OpNode) -> list[OpNode]:
        return [self.nodes[i] for i in node.inputs]

    def compute_nodes(self) -> list[OpNode]:
        return [n for n in self.nodes.values() if n.kind in ("compute", "layout")]

    def value(self, name: str) -> torch.Tensor:
        """Concrete tensor of a param/const node."""
        return self.params[name]

    def is_constant(self, name: str) -> bool:
        return self.nodes[name].kind in ("param", "const")

    # ---------------------------------------------------------------- rewrite
    def replace_all_uses(self, old: str, new: str) -> None:
        if old == new:
            return
        for n in self.nodes.values():
            n.inputs = [new if i == old else i for i in n.inputs]
        self.outputs = [new if o == old else o for o in self.outputs]

    def remove(self, name: str) -> None:
        if self.num_uses(name):
            raise ValueError(f"cannot remove {name}: still used")
        del self.nodes[name]
        self.params.pop(name, None)
        if name in self.inputs:
            self.inputs.remove(name)

    def dce(self) -> int:
        """Drop nodes that do not reach an output. Inputs are kept."""
        live: set[str] = set(self.outputs)
        stack = list(self.outputs)
        while stack:
            n = self.nodes[stack.pop()]
            for i in n.inputs:
                if i not in live:
                    live.add(i)
                    stack.append(i)
        dead = [n for n in self.nodes if n not in live and self.nodes[n].kind != "input"]
        for n in dead:
            del self.nodes[n]
            self.params.pop(n, None)
        return len(dead)

    def toposort(self) -> None:
        """Re-order ``nodes`` so every producer precedes its users."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            for i in self.nodes[name].inputs:
                visit(i)
            order.append(name)

        for name in list(self.nodes):
            visit(name)
        self.nodes = {n: self.nodes[n] for n in order}

    def insert_before(self, anchor: str, node: OpNode) -> None:
        """Place an already-added ``node`` right before ``anchor`` in order."""
        names = [n for n in self.nodes if n != node.name]
        idx = names.index(anchor)
        names.insert(idx, node.name)
        self.nodes = {n: self.nodes[n] for n in names}

    # ----------------------------------------------------------------- output
    def summary(self) -> str:
        lines = [f"OpGraph {self.name}: {len(self.nodes)} nodes, inputs={self.inputs}, outputs={self.outputs}"]
        for n in self.nodes.values():
            if n.kind in ("param", "const"):
                lines.append(f"  [{n.kind}] {n.name}: {n.type}")
            else:
                lines.append(f"  {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": _jsonable(self.metadata),
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OpGraph":
        g = OpGraph(d.get("name", "graph"))
        for nd in d["nodes"]:
            node = OpNode.from_dict(nd)
            g.nodes[node.name] = node
        g.inputs = list(d.get("inputs", []))
        g.outputs = list(d.get("outputs", []))
        g.metadata = dict(d.get("metadata", {}))
        return g

    @staticmethod
    def from_json(path: str | Path) -> "OpGraph":
        return OpGraph.from_dict(json.loads(Path(path).read_text()))

    def save_params(self, path: str | Path) -> None:
        torch.save({k: v for k, v in self.params.items()}, path)

    def copy(self) -> "OpGraph":
        g = OpGraph.from_dict(self.to_dict())
        g.params = dict(self.params)
        g._counter = self._counter
        return g


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
