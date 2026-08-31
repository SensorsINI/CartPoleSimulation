#!/usr/bin/env python3
"""Emit firmware RPGD config + golden vectors from the host C library."""
import argparse
import ctypes
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from others.Tests.validate_rpgd_c_tier_b import (
    RpgdRuntime,
    build_config,
    configure_c_api,
    load_c_lib,
)


FIRMWARE_GENERAL = ROOT.parent.parent / "Firmware" / "Src" / "General"
FIRMWARE_APP = ROOT.parent.parent / "Firmware" / "Src" / "CartPoleFirmware"
INT_FIELDS = {
    "mpc_horizon",
    "num_rollouts",
    "outer_its",
    "resamp_per",
    "period_interpolation_inducing_points",
    "intermediate_steps",
    "shift_previous",
    "sampling_distribution",
    "sample_whole_control_space",
    "warmup",
    "warmup_iterations",
    "num_threads",
    "reserve_threads",
    "seed",
}


def c_float(value):
    text = f"{float(np.float32(value)):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def c_int(name, value):
    if name == "seed":
        return f"{int(value)}u"
    return str(int(value))


def write_config_header(cfg, path, fingerprint):
    fields = [name for name, _ in cfg._fields_]
    lines = [
        "/* Generated from Control_Toolkit_ASF YAML + CartPoleParameters.",
        " * Do not edit by hand; rerun generate_rpgd_zynq_golden.py",
        " */",
        f"/* Source/config fingerprint: {fingerprint} */",
        "#ifndef RPGD_CONFIG_DEFAULTS_H",
        "#define RPGD_CONFIG_DEFAULTS_H",
        "",
        '#include "rpgd_c/rpgd_cartpole.h"',
        "",
        "#define RPGD_DEFAULT_CONFIG { \\",
    ]
    for name in fields:
        value = 1 if name == "num_threads" else getattr(cfg, name)
        if name in INT_FIELDS:
            rendered = c_int(name, value)
        else:
            rendered = c_float(value)
        comma = "," if name != fields[-1] else ""
        lines.append(f"    .{name} = {rendered}{comma} \\")
    lines.extend(["}", "", "#endif", ""])
    path.write_text("\n".join(lines))


def fmt_array(name, values, ctype="float"):
    rendered = [
        c_float(v) if ctype == "float" else str(int(v))
        for v in values
    ]
    lines = [f"static const {ctype} {name}[] = {{"]
    for start in range(0, len(rendered), 8):
        lines.append("    " + ", ".join(rendered[start:start + 8]) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_golden_header(path, state, q_init, q_kernel, u_steps, costs, indices, grad, fingerprint):
    lines = [
        "/* Generated golden vectors from the host RPGD-C library (seed 123, q_init rng 1234).",
        " * On-target checks use tier-B absolute tolerances, not bit-exact ARM/x86 equality.",
        " */",
        f"/* Source/config fingerprint: {fingerprint} */",
        "#ifndef RPGD_GOLDEN_VECTORS_H",
        "#define RPGD_GOLDEN_VECTORS_H",
        "",
        "#define RPGD_GOLDEN_U_ABS_TOL      2.0e-4f",
        "#define RPGD_GOLDEN_COST_ABS_TOL   1.0e-3f",
        "#define RPGD_GOLDEN_GRAD_ABS_TOL   5.0e-4f",
        f"#define RPGD_GOLDEN_BEST_INDEX     {int(indices[0])}",
        f"#define RPGD_GOLDEN_N_STEPS        {len(u_steps)}",
        f"#define RPGD_GOLDEN_N_COSTS        {len(costs)}",
        f"#define RPGD_GOLDEN_HORIZON        {len(grad)}",
        f"#define RPGD_GOLDEN_Q_INIT_LEN     {q_init.size}",
        "",
        fmt_array("RPGD_GOLDEN_STATE", state),
        fmt_array("RPGD_GOLDEN_U_STEPS", u_steps),
        fmt_array("RPGD_GOLDEN_COSTS", costs),
        fmt_array("RPGD_GOLDEN_INDICES", indices, ctype="int"),
        fmt_array("RPGD_GOLDEN_GRAD", grad),
        fmt_array("RPGD_GOLDEN_Q_KERNEL", q_kernel),
        fmt_array("RPGD_GOLDEN_Q_INIT", q_init.reshape(-1)),
        "",
        '_Static_assert(RPGD_GOLDEN_Q_INIT_LEN == RPGD_GOLDEN_N_COSTS * RPGD_GOLDEN_HORIZON,',
        '               "RPGD golden Q dimensions do not match");',
        "",
        "#endif",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=None)
    args = parser.parse_args()
    cfg = build_config(num_threads=1)
    lib = configure_c_api(load_c_lib(args.lib)) if args.lib else load_c_lib()
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    q_kernel = np.array([0.05 * np.cos(i * 0.2) for i in range(cfg.mpc_horizon)], dtype=np.float32)
    rng = np.random.default_rng(1234)
    q_init = rng.uniform(-0.2, 0.2, size=(cfg.num_rollouts, cfg.mpc_horizon)).astype(np.float32)
    digest = hashlib.sha256(ctypes.string_at(ctypes.byref(cfg), ctypes.sizeof(cfg)))
    source_dir = ROOT / "Control_Toolkit" / "Optimizers" / "rpgd_c"
    for source_path in sorted((*source_dir.glob("*.c"), *source_dir.glob("*.h"))):
        digest.update(source_path.read_bytes())
    fingerprint = digest.hexdigest()[:16]

    c_state = (ctypes.c_float * 6)(*state)
    c_q = (ctypes.c_float * cfg.mpc_horizon)(*q_kernel)
    grad = np.empty((cfg.mpc_horizon,), dtype=np.float32)
    lib.rpgd_debug_gradient_adjoint(
        ctypes.byref(cfg),
        ctypes.byref(rt),
        c_state,
        c_q,
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    solver = lib.rpgd_create(ctypes.byref(cfg))
    if not solver:
        raise RuntimeError("Could not create RPGD solver for golden-vector generation")
    try:
        lib.rpgd_debug_set_q(solver, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        u_steps = []
        costs = np.empty((cfg.num_rollouts,), dtype=np.float32)
        indices = np.empty((cfg.num_rollouts,), dtype=np.int32)
        for i in range(3):
            u_steps.append(float(lib.rpgd_step(solver, c_state, ctypes.byref(rt))))
            if i == 0:
                lib.rpgd_debug_get_costs(solver, costs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                lib.rpgd_debug_get_indices(solver, indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    finally:
        lib.rpgd_destroy(solver)

    FIRMWARE_GENERAL.mkdir(parents=True, exist_ok=True)
    FIRMWARE_APP.mkdir(parents=True, exist_ok=True)
    write_config_header(cfg, FIRMWARE_GENERAL / "rpgd_config_defaults.h", fingerprint)
    write_golden_header(
        FIRMWARE_APP / "rpgd_golden_vectors.h",
        state,
        q_init,
        q_kernel,
        u_steps,
        costs,
        indices,
        grad,
        fingerprint,
    )
    print(f"wrote {FIRMWARE_GENERAL / 'rpgd_config_defaults.h'}")
    print(f"wrote {FIRMWARE_APP / 'rpgd_golden_vectors.h'}")
    print(f"golden_u0={u_steps[0]:.9g} best={int(indices[0])}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
