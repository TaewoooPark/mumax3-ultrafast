#!/usr/bin/env python3
"""Generate MuMax3's Darwin Metal wrappers and combined MSL source.

This translator intentionally accepts only the small CUDA dialect audited in
cuda/*.cu. It fails closed when a source adds an unsupported parameter type or
CUDA construct, so a physics kernel can never be silently mistranslated.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

EXPECTED_KERNEL_COUNT = 65

KERNEL_SIGNATURE_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+'
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*?)\)\s*\{",
    re.DOTALL,
)
ARGUMENT_RE = re.compile(
    r"^(?P<base>float|int|uint8_t)\s*(?P<pointer>\*)?\s*"
    r"(?P<name>[A-Za-z_]\w*)$"
)
INCLUDE_RE = re.compile(r"^[ \t]*#include[^\n]*(?:\n|$)", re.MULTILINE)
LOCAL_MACRO_RE = re.compile(
    r"^[ \t]*#define[ \t]+(?P<name>[A-Za-z_]\w*)", re.MULTILINE
)
DEVICE_HELPER_RE = re.compile(
    r"(?:__device__\s+inline|inline\s+__device__)\s+"
    r"[A-Za-z_]\w*\s+(?P<name>[A-Za-z_]\w*)\s*\("
)

ARGUMENT_KINDS = {
    "float*": "buffer_float32",
    "uint8_t*": "buffer_uint8",
    "float": "float32",
    "int": "int32",
    "uint8_t": "uint8",
}

UNSUPPORTED_CONSTRUCTS = (
    "__global__",
    "__shared__",
    "__syncthreads",
    "__constant__",
    "__host__",
    "__launch_bounds__",
    "__shfl",
    "__ballot",
)


class GenerationError(RuntimeError):
    """Raised when the CUDA corpus no longer matches the audited dialect."""


@dataclasses.dataclass(frozen=True)
class Argument:
    name: str
    c_type: str
    kind: str
    buffer_index: int
    nullable: bool

    def public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "c_type": self.c_type,
            "kind": self.kind,
            "buffer_index": self.buffer_index,
            "nullable": self.nullable,
        }


@dataclasses.dataclass(frozen=True)
class Kernel:
    source: str
    name: str
    arguments: tuple[Argument, ...]
    source_sha256: str
    original_source: str
    local_macro_names: tuple[str, ...]

    @property
    def uses_pointer_mask(self) -> bool:
        """True when some argument is a NULL-sentinel CUDA pointer.

        Only those kernels read mumaxPointerMask, so binding it everywhere
        costs a setBytes call and a buffer-table slot on every dispatch of the
        other kernels.
        """
        return any(argument.nullable for argument in self.arguments)

    def public_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "name": self.name,
            "arguments": [argument.public_dict() for argument in self.arguments],
            "pointer_mask_buffer_index": (
                len(self.arguments) if self.uses_pointer_mask else None
            ),
            "source_sha256": self.source_sha256,
        }


def split_arguments(text: str) -> list[str]:
    """Split the flat audited argument grammar, rejecting nested surprises."""

    depth = 0
    start = 0
    result: list[str] = []
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth < 0:
                raise GenerationError("unbalanced delimiter in kernel arguments")
        elif character == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        raise GenerationError("unbalanced delimiter in kernel arguments")
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    if not result:
        raise GenerationError("kernel has no arguments")
    return result


def parse_kernel(path: Path) -> Kernel:
    source = path.read_text(encoding="utf-8")
    matches = list(KERNEL_SIGNATURE_RE.finditer(source))
    if len(matches) != 1:
        raise GenerationError(
            f"{path.name}: expected exactly one extern C global kernel, "
            f"found {len(matches)}"
        )
    match = matches[0]
    name = match.group("name")
    parsed_arguments: list[tuple[str, str, str, int]] = []
    for buffer_index, raw_argument in enumerate(
        split_arguments(match.group("args"))
    ):
        normalized = " ".join(
            raw_argument.replace("__restrict__", "").split()
        )
        argument_match = ARGUMENT_RE.fullmatch(normalized)
        if argument_match is None:
            raise GenerationError(
                f"{path.name}/{name}: unsupported argument {normalized!r}"
            )
        c_type = argument_match.group("base")
        if argument_match.group("pointer"):
            c_type += "*"
        try:
            kind = ARGUMENT_KINDS[c_type]
        except KeyError as error:
            raise GenerationError(
                f"{path.name}/{name}: unsupported CUDA type {c_type!r}"
            ) from error
        parsed_arguments.append(
            (argument_match.group("name"), c_type, kind, buffer_index)
        )

    nullable_names = find_nullable_pointer_names(source)
    arguments = tuple(
        Argument(
            name=argument_name,
            c_type=c_type,
            kind=kind,
            buffer_index=buffer_index,
            nullable=argument_name in nullable_names,
        )
        for argument_name, c_type, kind, buffer_index in parsed_arguments
    )
    declared_pointer_names = {
        argument.name for argument in arguments if argument.c_type.endswith("*")
    }
    undeclared_nullable = nullable_names - declared_pointer_names
    if undeclared_nullable:
        raise GenerationError(
            f"{path.name}/{name}: nullable-pointer analysis found unknown "
            f"arguments {sorted(undeclared_nullable)}"
        )

    return Kernel(
        source=path.name,
        name=name,
        arguments=arguments,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        original_source=source,
        local_macro_names=tuple(
            macro.group("name") for macro in LOCAL_MACRO_RE.finditer(source)
        ),
    )


def find_nullable_pointer_names(source: str) -> set[str]:
    """Find pointer arguments whose NULL sentinel has CUDA scalar semantics."""

    names: set[str] = set()
    for match in re.finditer(
        r"\b(?:amul|inv_Msat)\s*\(\s*([A-Za-z_]\w*)\s*,", source
    ):
        names.add(match.group(1))
    for match in re.finditer(
        r"\bvmul\s*\(\s*([A-Za-z_]\w*)\s*,\s*"
        r"([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,",
        source,
    ):
        names.update(match.groups())
    for match in re.finditer(
        r"\b([A-Za-z_]\w*)\s*(?:==|!=)\s*NULL\b", source
    ):
        names.add(match.group(1))
    for match in re.finditer(
        r"\bNULL\s*(?:==|!=)\s*([A-Za-z_]\w*)\b", source
    ):
        names.add(match.group(1))
    return names


def load_kernels(cuda_dir: Path) -> list[Kernel]:
    paths = sorted(cuda_dir.glob("*.cu"))
    kernels = [parse_kernel(path) for path in paths]
    if len(kernels) != EXPECTED_KERNEL_COUNT:
        raise GenerationError(
            "audited production kernel count changed: "
            f"got {len(kernels)}, want {EXPECTED_KERNEL_COUNT}; "
            "audit new or removed kernels before regenerating"
        )
    seen: dict[str, str] = {}
    for kernel in kernels:
        if kernel.name in seen:
            raise GenerationError(
                f"duplicate kernel {kernel.name!r} in "
                f"{seen[kernel.name]} and {kernel.source}"
            )
        seen[kernel.name] = kernel.source
    return kernels


def sanitize_identifier(value: str) -> str:
    result = []
    for index, character in enumerate(value):
        if (
            character == "_"
            or character.isascii()
            and character.isalpha()
            or index > 0
            and character.isascii()
            and character.isdigit()
        ):
            result.append(character)
        else:
            result.append("_")
    return "".join(result)


def metal_argument_type(c_type: str) -> str:
    return {
        "float*": "device float*",
        "uint8_t*": "device uchar*",
        "float": "constant float&",
        "int": "constant int&",
        "uint8_t": "constant uchar&",
    }[c_type]


def metal_signature(kernel: Kernel, builtins: tuple[str, ...] = None) -> str:
    if builtins is None:
        # The reduction macro also collapses across SIMD lanes for the
        # maximum reductions, so it needs the lane and group indices.
        builtins = (
            "    uint3 blockIdx [[threadgroup_position_in_grid]],",
            "    uint3 threadIdx [[thread_position_in_threadgroup]],",
            "    uint3 blockDim [[threads_per_threadgroup]],",
            "    uint3 gridDim [[threadgroups_per_grid]],",
            "    uint mumaxSimdLane [[thread_index_in_simdgroup]],",
            "    uint mumaxSimdGroup [[simdgroup_index_in_threadgroup]]) {",
        )
    lines = [f"kernel void {kernel.name}("]
    for argument in kernel.arguments:
        lines.append(
            f"    {metal_argument_type(argument.c_type)} {argument.name} "
            f"[[buffer({argument.buffer_index})]],"
        )
    if kernel.uses_pointer_mask:
        lines.append(
            f"    constant uint& mumaxPointerMask "
            f"[[buffer({len(kernel.arguments)})]],"
        )
    lines.extend(builtins)
    return "\n".join(lines)


# CUDA recomputes a global thread index from four launch built-ins. Metal
# publishes it directly as thread_position_in_grid, so materialising
# threadgroup_position_in_grid, thread_position_in_threadgroup,
# threads_per_threadgroup and threadgroups_per_grid only burns registers and
# integer MADs on every one of the ~115 dispatches in an RK45DP step.
#
# Both rewrites below are exact identities, not approximations:
#
#   3D: blockIdx.<a>*blockDim.<a> + threadIdx.<a> is the definition of
#       thread_position_in_grid.<a>.
#
#   1D: (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x
#       == gid.y*(gridDim.x*blockDim.x) + gid.x
#       == gid.y*threads_per_grid.x + gid.x
#       given blockDim.y == 1, which every call site of the 1D form uses. The
#       original expression drops threadIdx.y, so it is only correct under that
#       same condition.
GLOBAL_INDEX_RE = re.compile(
    r"int\s+(?P<name>[A-Za-z_]\w*)\s*=\s*\(\s*blockIdx\.y\s*\*\s*gridDim\.x"
    r"\s*\+\s*blockIdx\.x\s*\)\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;"
)
AXIS_INDEX_RE = re.compile(
    r"int\s+(?P<name>[A-Za-z_]\w*)\s*=\s*blockIdx\.(?P<axis>[xyz])\s*\*"
    r"\s*blockDim\.(?P=axis)\s*\+\s*threadIdx\.(?P=axis)\s*;"
)


def rewrite_thread_indices(source: str) -> str:
    """Replace CUDA global-index arithmetic with Metal's native built-ins."""

    source = GLOBAL_INDEX_RE.sub(
        lambda m: (
            f"int {m.group('name')} = "
            "int(mumaxGid.y * mumaxThreadsPerGrid.x + mumaxGid.x);"
        ),
        source,
    )
    source = AXIS_INDEX_RE.sub(
        lambda m: f"int {m.group('name')} = int(mumaxGid.{m.group('axis')});",
        source,
    )
    return source


CUDA_LAUNCH_BUILTINS = ("blockIdx", "threadIdx", "blockDim", "gridDim")

# Prelude macros that expand to CUDA launch geometry. A kernel whose body only
# invokes such a macro shows no built-in token of its own, so it must be matched
# by name or its signature would omit declarations the expansion needs. The
# reduction macro strides by gridDim.x*blockDim.x and indexes sdata by
# threadIdx.x, so it cannot move to thread_position_in_grid without changing the
# reduction itself.
LAUNCH_GEOMETRY_MACROS = ("reduce",)


def translate_kernel(kernel: Kernel) -> str:
    source = INCLUDE_RE.sub("", kernel.original_source)

    # CUDA translation units may reuse file-local helper names. Metal receives
    # one combined library, so namespace every helper by source stem.
    stem = sanitize_identifier(Path(kernel.source).stem)
    helper_names = {
        match.group("name") for match in DEVICE_HELPER_RE.finditer(source)
    }
    for helper_name in sorted(helper_names):
        source = re.sub(
            rf"\b{re.escape(helper_name)}\b",
            f"{helper_name}__{stem}",
            source,
        )

    source = translate_nullable_pointer_operations(source, kernel)
    source = rewrite_thread_indices(source)

    matches = list(KERNEL_SIGNATURE_RE.finditer(source))
    if len(matches) != 1:
        raise GenerationError(
            f"{kernel.source}/{kernel.name}: signature disappeared "
            "during translation"
        )
    match = matches[0]
    body = source[match.end():]
    invokes_geometry_macro = any(
        re.search(rf"\b{re.escape(name)}\s*\(", body)
        for name in LAUNCH_GEOMETRY_MACROS
    )
    if invokes_geometry_macro or any(
        builtin in body for builtin in CUDA_LAUNCH_BUILTINS
    ):
        # Reduction kernels stride by gridDim.x*blockDim.x, so they still need
        # the CUDA launch geometry.
        builtins = None
    else:
        declarations = ["    uint3 mumaxGid [[thread_position_in_grid]],"]
        if "mumaxThreadsPerGrid" in body:
            declarations.append(
                "    uint3 mumaxThreadsPerGrid [[threads_per_grid]],"
            )
        declarations[-1] = declarations[-1][:-1] + ") {"
        builtins = tuple(declarations)
    source = (
        source[: match.start()]
        + metal_signature(kernel, builtins)
        + source[match.end() :]
    )

    token_replacements = {
        "__device__": "",
        "__restrict__": "",
        "uint8_t": "uchar",
        "NULL": "nullptr",
        "sqrtf": "sqrt",
        "acosf": "acos",
        "floorf": "floor",
        "fmodf": "fmod",
        "atan2f": "atan2",
        "cuDoubleComplex": "float2",
        "make_cuDoubleComplex": "mumaxMakeComplex",
        "cuCadd": "mumaxComplexAdd",
        "cuCsub": "mumaxComplexSub",
        "cuCmul": "mumaxComplexMul",
        "cuCimag": "mumaxComplexImag",
    }
    for old, new in token_replacements.items():
        source = re.sub(rf"\b{re.escape(old)}\b", new, source)

    for construct in UNSUPPORTED_CONSTRUCTS:
        if construct in source:
            raise GenerationError(
                f"{kernel.source}/{kernel.name}: unsupported CUDA construct "
                f"remains after translation: {construct}"
            )
    if re.search(r"\b(?:cu[A-Z]\w*|make_cu\w*)\b", source):
        raise GenerationError(
            f"{kernel.source}/{kernel.name}: an unsupported CUDA library "
            "symbol remains after translation"
        )
    return source


def translate_nullable_pointer_operations(
    source: str, kernel: Kernel
) -> str:
    """Route CUDA NULL-sentinel operations through the explicit ABI mask."""

    argument_by_name = {argument.name: argument for argument in kernel.arguments}

    for argument in kernel.arguments:
        if not argument.nullable:
            continue
        present = (
            f"mumaxPointerPresent(mumaxPointerMask, "
            f"{argument.buffer_index}u)"
        )
        name = re.escape(argument.name)
        source = re.sub(
            rf"\b{argument.name}\s*==\s*NULL\b", f"!{present}", source
        )
        source = re.sub(
            rf"\b{argument.name}\s*!=\s*NULL\b", present, source
        )
        source = re.sub(
            rf"\bNULL\s*==\s*{name}\b", f"!{present}", source
        )
        source = re.sub(
            rf"\bNULL\s*!=\s*{name}\b", present, source
        )
        for function in ("amul", "inv_Msat"):
            source = re.sub(
                rf"\b{function}\s*\(\s*{name}\s*,",
                f"{function}({argument.name}, {present},",
                source,
            )

    vmul_pattern = re.compile(
        r"\bvmul\s*\(\s*(?P<x>[A-Za-z_]\w*)\s*,\s*"
        r"(?P<y>[A-Za-z_]\w*)\s*,\s*(?P<z>[A-Za-z_]\w*)\s*,"
    )

    def replace_vmul(match: re.Match[str]) -> str:
        names = (match.group("x"), match.group("y"), match.group("z"))
        try:
            arguments = [argument_by_name[name] for name in names]
        except KeyError as error:
            raise GenerationError(
                f"{kernel.source}/{kernel.name}: vmul uses unknown pointer "
                f"{error.args[0]!r}"
            ) from error
        if not all(argument.nullable for argument in arguments):
            raise GenerationError(
                f"{kernel.source}/{kernel.name}: vmul pointer classification "
                f"is incomplete for {names}"
            )
        presence = [
            f"mumaxPointerPresent(mumaxPointerMask, "
            f"{argument.buffer_index}u)"
            for argument in arguments
        ]
        return (
            f"vmul({names[0]}, {names[1]}, {names[2]}, "
            f"{presence[0]}, {presence[1]}, {presence[2]},"
        )

    return vmul_pattern.sub(replace_vmul, source)


def verify_launch_builtins(msl: str) -> None:
    """Reject a library where a kernel uses a built-in it does not declare."""

    blocks = re.split(r"(?m)^kernel void ", msl)
    prelude_text, kernel_blocks = blocks[0], blocks[1:]
    geometry_macro_bodies = {
        name
        for name in LAUNCH_GEOMETRY_MACROS
        if any(builtin in prelude_text for builtin in CUDA_LAUNCH_BUILTINS)
        and re.search(rf"(?m)^#define\s+{re.escape(name)}\b", prelude_text)
    }
    for block in kernel_blocks:
        name = block.split("(", 1)[0].strip()
        signature_end = block.index("{")
        signature, body = block[:signature_end], block[signature_end:]
        declares_cuda = all(
            builtin in signature for builtin in CUDA_LAUNCH_BUILTINS
        )
        if declares_cuda:
            continue
        for builtin in CUDA_LAUNCH_BUILTINS:
            if builtin in body:
                raise GenerationError(
                    f"{name}: uses {builtin} but does not declare the CUDA "
                    "launch built-ins"
                )
        for macro in geometry_macro_bodies:
            if re.search(rf"\b{re.escape(macro)}\s*\(", body):
                raise GenerationError(
                    f"{name}: invokes the {macro} macro, which expands to CUDA "
                    "launch geometry, but does not declare the built-ins"
                )


def generate_msl(kernels: Iterable[Kernel], prelude: str) -> str:
    chunks = [prelude.rstrip(), ""]
    for kernel in kernels:
        chunks.extend(
            (
                "// "
                "-----------------------------------------------------------------------------",
                f"// Source: cuda/{kernel.source}",
                "// "
                "-----------------------------------------------------------------------------",
                f'#line 1 "{kernel.source}"',
                translate_kernel(kernel).rstrip(),
            )
        )
        chunks.extend(f"#undef {name}" for name in kernel.local_macro_names)
        chunks.append("")
    combined = "\n".join(chunks)
    document = "\n".join(line.rstrip() for line in combined.splitlines()) + "\n"
    verify_launch_builtins(document)
    return document


def go_argument_type(c_type: str) -> str:
    return {
        "float*": "unsafe.Pointer",
        "uint8_t*": "unsafe.Pointer",
        "float": "float32",
        "int": "int",
        "uint8_t": "byte",
    }[c_type]


def constructor_for(c_type: str) -> str:
    return {
        "float*": "metal.BufferArg",
        "uint8_t*": "metal.BufferArg",
        "float": "metal.F32",
        "int": "metal.I32",
        "uint8_t": "metal.U8",
    }[c_type]


def generate_wrapper(kernel: Kernel) -> str:
    lines = [
        "// Code generated by cuda/metal/cmd/cuda2metal; DO NOT EDIT.",
        "//go:build darwin && arm64",
        "",
        "package cuda",
        "",
        "import (",
        '\t"unsafe"',
        "",
        '\t"github.com/mumax/3/cuda/metal"',
        '\t"github.com/mumax/3/timer"',
        ")",
        "",
        f"// kernel_{kernel.name} caches the runtime handle so each dispatch "
        f"skips the kernel-name lookup.",
        f'var kernel_{kernel.name} = metal.NewKernel("{kernel.name}")',
        "",
        f"// k_{kernel.name}_async dispatches the Metal implementation "
        f"of cuda/{kernel.source}.",
        f"func k_{kernel.name}_async(",
    ]
    lines.extend(
        f"\t{argument.name} {go_argument_type(argument.c_type)},"
        for argument in kernel.arguments
    )
    lines.extend(
        (
            "\tcfg *config,",
            ") {",
            "\tif Synchronous {",
            "\t\tSync()",
            f'\t\ttimer.Start("{kernel.name}")',
            "\t}",
            "",
        )
    )
    for argument in kernel.arguments:
        if argument.c_type.endswith("*") and not argument.nullable:
            lines.extend(
                (
                    f"\tif {argument.name} == nil {{",
                    f'\t\tpanic("cuda/metal: kernel {kernel.name} argument '
                    f'{argument.name} must not be nil")',
                    "\t}",
                )
            )
    if kernel.uses_pointer_mask:
        lines.extend(("", "\tvar mumaxPointerMask uint32"))
        for argument in kernel.arguments:
            if argument.c_type.endswith("*"):
                lines.extend(
                    (
                        f"\tif {argument.name} != nil {{",
                        f"\t\tmumaxPointerMask |= uint32(1) << "
                        f"{argument.buffer_index}",
                        "\t}",
                    )
                )
    lines.extend(
        (
            "",
            f"\tkernel_{kernel.name}.MustLaunch(metal.GridConfig{{",
            "\t\tGridX: uint32(cfg.Grid.X), GridY: uint32(cfg.Grid.Y), "
            "GridZ: uint32(cfg.Grid.Z),",
            "\t\tBlockX: uint32(cfg.Block.X), BlockY: uint32(cfg.Block.Y), "
            "BlockZ: uint32(cfg.Block.Z),",
            "\t\tThreadsX: uint32(cfg.Threads.X), ThreadsY: uint32(cfg.Threads.Y), "
            "ThreadsZ: uint32(cfg.Threads.Z),",
            "\t},",
        )
    )
    lines.extend(
        f"\t\t{constructor_for(argument.c_type)}({argument.name}),"
        for argument in kernel.arguments
    )
    if kernel.uses_pointer_mask:
        lines.append("\t\tmetal.U32(mumaxPointerMask),")
    lines.extend(
        (
            "\t)",
            "",
            "\tif Synchronous {",
            "\t\tSync()",
            f'\t\ttimer.Stop("{kernel.name}")',
            "\t}",
            "}",
            "",
        )
    )
    return "\n".join(lines)


def manifest_text(kernels: Iterable[Kernel]) -> str:
    kernel_list = list(kernels)
    document = {
        "schema_version": 1,
        "kernel_count": len(kernel_list),
        "kernels": [kernel.public_dict() for kernel in kernel_list],
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def output_map(
    kernels: Iterable[Kernel],
    prelude: str,
    out_dir: Path,
    wrapper_dir: Path,
) -> dict[Path, str]:
    kernel_list = list(kernels)
    outputs = {
        out_dir / "mumax3_kernels.metal": generate_msl(kernel_list, prelude),
        out_dir / "manifest.json": manifest_text(kernel_list),
    }
    for kernel in kernel_list:
        stem = Path(kernel.source).stem
        outputs[wrapper_dir / f"{stem}_wrapper_metal.go"] = generate_wrapper(
            kernel
        )
    return outputs


def write_outputs(outputs: dict[Path, str]) -> None:
    for path in sorted(outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], encoding="utf-8")


def check_outputs(outputs: dict[Path, str]) -> None:
    stale: list[str] = []
    for path in sorted(outputs):
        if not path.exists():
            stale.append(f"{path} (missing)")
        elif path.read_text(encoding="utf-8") != outputs[path]:
            stale.append(f"{path} (stale)")
    if stale:
        raise GenerationError(
            "generated Metal files are not current:\n  " + "\n  ".join(stale)
        )


def default_paths() -> tuple[Path, Path, Path, Path]:
    script = Path(__file__).resolve()
    metal_dir = script.parents[2]
    repo_root = script.parents[4]
    return (
        repo_root / "cuda",
        metal_dir / "kernels" / "compatibility.metal",
        metal_dir / "kernels",
        repo_root / "cuda",
    )


def parse_cli(argv: list[str]) -> argparse.Namespace:
    cuda_dir, prelude, out_dir, wrapper_dir = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, default=cuda_dir)
    parser.add_argument("--prelude", type=Path, default=prelude)
    parser.add_argument("--out-dir", type=Path, default=out_dir)
    parser.add_argument("--wrapper-dir", type=Path, default=wrapper_dir)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in generated files instead of writing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(sys.argv[1:] if argv is None else argv)
    try:
        kernels = load_kernels(args.cuda_dir)
        prelude = args.prelude.read_text(encoding="utf-8")
        outputs = output_map(kernels, prelude, args.out_dir, args.wrapper_dir)
        if args.check:
            check_outputs(outputs)
        else:
            write_outputs(outputs)
    except (GenerationError, OSError) as error:
        print(f"cuda2metal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
