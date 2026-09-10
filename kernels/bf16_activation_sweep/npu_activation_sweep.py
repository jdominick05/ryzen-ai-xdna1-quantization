#!/usr/bin/env python3
"""bf16 activation kernels (ReLU / SiLU / GELU) on the NPU against Zen 4 torch bf16.

The backlog item this answers ("BF16 Continuous Activation Engine") asks for vector bf16
activation kernels on AIE vector units. mlir-aie already ships them and this machine has
already run them -- `results/aie/mlir_aie_ml_examples_npu.log` records relu/silu/gelu all
`PASS!` -- but nothing was ever TIMED, so there was no verdict, only a correctness check.

This drives the upstream `ml/eltwise_unary` design (v1.4.2, unmodified) through its own
`@iron.jit` entry point and times it with mlir-aie's own `run_iters`, then times the same
op on the CPU with torch bf16 in the SAME PROCESS -- ironenv has torch 2.14.0+cpu, so both
legs share one sitting and CLAUDE.md's cross-session drift invariant is satisfied by
construction rather than by promise.

An activation is pure bandwidth: 2 bytes in, 2 bytes out, one cheap op per element, no
reuse. So the interesting number is not GOPS, it is whether the transfer can pay for a
dispatch at all. This repo has already measured the host-set dispatch floor -- 617 us
one-shot through @iron.jit, ~531 us batched IRON, 36.7 us batched C++ (see
`results/aie/dispatch_floor_npu.log`) -- and an activation only wins if the CPU's time for
the same tensor exceeds that. The sweep exists to find where, or to show there is no such
point.

Accuracy caveat, and it is the design's own: relu is exact, but silu and gelu are LUT
approximations that mlir-aie itself verifies at rtol=0.128 (12.8%) and, for gelu,
atol=0.05. That is a wide tolerance for a kernel proposed as a replacement for a vendor
quantizer's activation handling, so this script reports the observed max relative error
per row rather than only PASS/FAIL.

    python kernels/bf16_activation_sweep/npu_activation_sweep.py \
        --ops relu,silu,gelu --lengths 65536,262144,1048576,4194304 --iters 20 --warmup 5

Run from the mlir-aie ironenv (see kernels/README.md), NOT resnet_env17.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

DESIGN_DIR = (
    Path.home()
    / "mlir-aie"
    / "programming_examples"
    / "ml"
    / "eltwise_unary"
)


def _load_design():
    """Import the upstream design module without running its CLI."""
    if not DESIGN_DIR.is_dir():
        sys.exit(f"design dir not found: {DESIGN_DIR}")
    sys.path.insert(0, str(DESIGN_DIR))
    import eltwise_unary as design  # noqa: E402

    return design


def cpu_time_us(fn, x, warmup, iters):
    """Median-of-iters wall time in microseconds for fn(x), after warmup."""
    for _ in range(warmup):
        fn(x)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples), min(samples)


# mlir-aie's own tolerances for these kernels, from ml/eltwise_unary's _VERIFY_CFG.
# relu is exact; silu and gelu are LUT approximations.
_TOL = {
    "relu": dict(rtol=0.0, atol=0.0),
    "silu": dict(rtol=0.128, atol=0.0),
    "gelu": dict(rtol=0.128, atol=0.05),
}


def accuracy(got, ref, op):
    """Report error the way the design's own check does, plus a plain abs error.

    A bare max-relative-error is misleading for gelu: its reference is a small negative
    number in the tail (gelu(-3) ~ -0.0036), so a kernel that returns 0 there scores a
    relative error of exactly 1.0 while being off by 0.0036 in absolute terms. That is
    why upstream pairs rtol with atol for gelu, and why both are reported here.
    """
    import numpy as np

    g = got.astype(np.float32)
    r = ref.astype(np.float32)
    tol = _TOL[op]
    abs_err = np.abs(g - r)
    max_abs = float(np.max(abs_err))
    # Relative error restricted to entries where the reference is large enough for a
    # ratio to mean anything; the tail is covered by max_abs and by the allclose verdict.
    big = np.abs(r) >= 0.1
    max_rel = float(np.max(abs_err[big] / np.abs(r[big]))) if big.any() else 0.0
    passes = bool(np.allclose(g, r, rtol=tol["rtol"], atol=tol["atol"]))
    return max_abs, max_rel, passes


def run_one(design, op, length, channels, args):
    import aie.iron as iron
    import numpy as np
    import torch
    import torch.nn.functional as F
    from aie.utils.benchmark import run_iters
    from ml_dtypes import bfloat16

    rng = np.random.default_rng(0)
    in_np = rng.uniform(-3.0, 3.0, size=(length,)).astype(bfloat16)

    a_t = iron.tensor(in_np, dtype=bfloat16, device="npu")
    b_t = iron.zeros_like(a_t)

    kwargs = dict(size=length, num_channels=channels, op=op)

    bench = run_iters(
        design.eltwise_unary,
        a_t,
        b_t,
        warmup=args.warmup,
        iters=args.iters,
        **kwargs,
    )

    out = b_t.numpy()
    from aie.iron import kernels as aie_kernels

    ref_fn = {
        "relu": aie_kernels.relu_ref,
        "silu": aie_kernels.silu_ref,
        "gelu": aie_kernels.gelu_ref,
    }[op]
    ref = ref_fn(in_np)
    max_abs, max_rel, passes = accuracy(out, ref, op)

    # CPU leg, same process, same sitting, same data.
    #
    # MORE THAN ONE CPU KERNEL PER OP, deliberately. This repo has twice reached a wrong
    # verdict by timing the slower of two available CPU implementations
    # (conv2x_int8_cpu_baseline.log, the MobileViT splice), and int8_matmul_sweep.py
    # answers that by timing two int8 GEMMs and letting the faster one be the verdict.
    # The same rule applies here, and it matters: torch's bf16 SiLU turned out ~17x
    # slower than its own bf16 ReLU on the same tensor, which is exactly the shape of an
    # unoptimized fallback path. The verdict line is the FASTEST CPU kernel found.
    x32 = torch.from_numpy(in_np.astype(np.float32))
    x16 = x32.to(torch.bfloat16)

    variants = {
        "relu": {
            "torch bf16": (F.relu, x16),
            "torch fp32": (F.relu, x32),
            "clamp bf16": (lambda t: torch.clamp(t, min=0), x16),
        },
        "silu": {
            "torch bf16": (F.silu, x16),
            "torch fp32": (F.silu, x32),
            "x*sigmoid bf16": (lambda t: t * torch.sigmoid(t), x16),
            "x*sigmoid fp32": (lambda t: t * torch.sigmoid(t), x32),
        },
        "gelu": {
            "torch bf16": (F.gelu, x16),
            "torch fp32": (F.gelu, x32),
            "tanh bf16": (lambda t: F.gelu(t, approximate="tanh"), x16),
        },
    }[op]

    cpu_results = {}
    for name, (fn, tensor) in variants.items():
        med, mn = cpu_time_us(fn, tensor, args.warmup, args.iters)
        cpu_results[name] = (med, mn)

    cpu_best_name = min(cpu_results, key=lambda k: cpu_results[k][0])
    cpu_med, cpu_min = cpu_results[cpu_best_name]

    npu_avg = bench.npu.avg_us if bench.npu else float("nan")
    npu_min = bench.npu.min_us if bench.npu else float("nan")
    e2e_avg = bench.e2e.avg_us

    bytes_moved = 4 * length  # 2 in + 2 out
    npu_gbs = bytes_moved / (npu_avg * 1e-6) / 1e9 if npu_avg == npu_avg else 0.0
    cpu_gbs = bytes_moved / (cpu_med * 1e-6) / 1e9

    return dict(
        op=op,
        length=length,
        npu_avg=npu_avg,
        npu_min=npu_min,
        e2e_avg=e2e_avg,
        cpu_med=cpu_med,
        cpu_min=cpu_min,
        npu_gbs=npu_gbs,
        cpu_gbs=cpu_gbs,
        max_abs=max_abs,
        max_rel=max_rel,
        passes=passes,
        cpu_best_name=cpu_best_name,
        cpu_results=cpu_results,
        ratio_npu=cpu_med / npu_avg if npu_avg else float("nan"),
        ratio_e2e=cpu_med / e2e_avg if e2e_avg else float("nan"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ops", default="relu,silu,gelu")
    p.add_argument(
        "--lengths",
        default="65536,262144,1048576,4194304",
        help="element counts, comma separated",
    )
    p.add_argument("--channels", type=int, default=2, choices=(1, 2))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    design = _load_design()

    import torch

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]

    print(
        f"bf16 activation sweep  channels={args.channels}  "
        f"iters={args.iters} warmup={args.warmup}  torch={torch.__version__} "
        f"threads={torch.get_num_threads()}"
    )
    print(f"design: {DESIGN_DIR}")

    rows = []
    for op in ops:
        for length in lengths:
            print(f"\n=== {op}  L={length}  ch={args.channels} ===", flush=True)
            try:
                r = run_one(design, op, length, args.channels, args)
            except Exception as exc:  # a shape the design refuses is a result too
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                rows.append(dict(op=op, length=length, failed=str(exc)))
                continue
            print(
                f"  NPU  avg {r['npu_avg']:9.1f} us  min {r['npu_min']:9.1f} us"
                f"   {r['npu_gbs']:6.2f} GB/s"
            )
            print(f"  e2e  avg {r['e2e_avg']:9.1f} us   (host-side, includes dispatch)")
            for name, (med, mn) in sorted(
                r["cpu_results"].items(), key=lambda kv: kv[1][0]
            ):
                mark = "  <- verdict" if name == r["cpu_best_name"] else ""
                print(f"  CPU  {name:<16} med {med:9.1f} us  min {mn:9.1f} us{mark}")
            print(f"  CPU best {r['cpu_med']:9.1f} us   {r['cpu_gbs']:6.2f} GB/s")
            print(
                f"  CPU/NPU {r['ratio_npu']:6.2f}x on NPU time,"
                f" {r['ratio_e2e']:6.2f}x on e2e"
            )
            print(
                f"  accuracy: {'PASS' if r['passes'] else 'FAIL'} at the design's own"
                f" tolerance   max abs err {r['max_abs']:.5f}"
                f"   max rel err (|ref|>=0.1) {r['max_rel']:.5f}"
            )
            rows.append(r)

    print(
        "\nSUMMARY  (CPU/NPU > 1 means the NPU is faster; e2e is the honest column,"
        " it pays the dispatch)"
    )
    print(
        f"{'op':>5} {'L':>9} {'NPU us':>10} {'e2e us':>10} {'CPU us':>10} "
        f"{'NPU GB/s':>9} {'CPU GB/s':>9} {'CPU/NPU':>8} {'CPU/e2e':>8} "
        f"{'maxabs':>8} {'acc':>5}  CPU kernel"
    )
    print("-" * 124)
    for r in rows:
        if "failed" in r:
            print(f"{r['op']:>5} {r['length']:>9}   FAILED  {r['failed'][:60]}")
            continue
        print(
            f"{r['op']:>5} {r['length']:>9} {r['npu_avg']:>10.1f} {r['e2e_avg']:>10.1f} "
            f"{r['cpu_med']:>10.1f} {r['npu_gbs']:>9.2f} {r['cpu_gbs']:>9.2f} "
            f"{r['ratio_npu']:>8.2f} {r['ratio_e2e']:>8.2f} {r['max_abs']:>8.5f} "
            f"{'PASS' if r['passes'] else 'FAIL':>5}  {r['cpu_best_name']}"
        )


if __name__ == "__main__":
    main()
