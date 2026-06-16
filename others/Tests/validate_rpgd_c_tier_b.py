import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CartPole.cartpole_parameters import CartPoleParameters
from CartPole.state_utilities import (
    ANGLE_COS_IDX,
    ANGLE_IDX,
    ANGLE_SIN_IDX,
    ANGLED_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
    create_cartpole_state,
)
from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from Control_Toolkit.Optimizers.optimizer_rpgd_c import RpgdConfig, RpgdRuntime, optimizer_rpgd_c
from SI_Toolkit.load_and_normalize import load_yaml


def _scalar(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(np.asarray(value).reshape(-1)[0])


def build_config(num_threads=1):
    cp = CartPoleParameters()
    opt = load_yaml(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"))["rpgd_c"]
    cost = load_yaml(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))["CartPole"][
        "quadratic_boundary_grad_minimal"
    ]
    return RpgdConfig(
        opt["mpc_horizon"],
        opt["num_rollouts"],
        opt["outer_its"],
        opt["resamp_per"],
        opt["period_interpolation_inducing_points"],
        opt["intermediate_steps"],
        opt["shift_previous"],
        0,
        0,
        0,
        opt["warmup_iterations"],
        num_threads,
        1,
        123,
        opt["mpc_timestep"],
        opt["learning_rate"],
        opt["adam_beta_1"],
        opt["adam_beta_2"],
        opt["adam_epsilon"],
        opt["gradmax_clip"],
        opt["opt_keep_k_ratio"],
        opt["sample_stdev"],
        opt["sample_mean"],
        opt["uniform_dist_min"],
        opt["uniform_dist_max"],
        -1.0,
        1.0,
        _scalar(cp.k),
        _scalar(cp.m_cart),
        _scalar(cp.m_pole),
        _scalar(cp.g),
        _scalar(cp.J_fric),
        _scalar(cp.M_fric),
        _scalar(cp.L),
        _scalar(cp.u_max),
        _scalar(cp.TrackHalfLength),
        cost["dd_quadratic_weight_up"],
        cost["db_weight_up"],
        cost["ep_weight_up"],
        cost["ekp_weight_up"],
        cost["cc_weight_up"],
        cost["vel_penalty_reg"],
        cost["R"],
        cost["permissible_track_fraction"],
    )


def configure_c_api(lib):
    lib.rpgd_debug_rollout_cost.argtypes = [
        ctypes.POINTER(RpgdConfig),
        ctypes.POINTER(RpgdRuntime),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rpgd_debug_rollout_cost.restype = ctypes.c_float
    lib.rpgd_debug_rollout_final_state.argtypes = [
        ctypes.POINTER(RpgdConfig),
        ctypes.POINTER(RpgdRuntime),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rpgd_debug_gradient_adjoint.argtypes = [
        ctypes.POINTER(RpgdConfig),
        ctypes.POINTER(RpgdRuntime),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rpgd_debug_gradient_fd.argtypes = lib.rpgd_debug_gradient_adjoint.argtypes
    lib.rpgd_create.argtypes = [ctypes.POINTER(RpgdConfig)]
    lib.rpgd_create.restype = ctypes.c_void_p
    lib.rpgd_destroy.argtypes = [ctypes.c_void_p]
    lib.rpgd_debug_set_q.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rpgd_debug_get_costs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rpgd_debug_get_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.rpgd_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(RpgdRuntime),
    ]
    lib.rpgd_step.restype = ctypes.c_float
    return lib


def load_c_lib(path=None):
    if path is not None:
        return configure_c_api(ctypes.CDLL(str(path)))
    c_dir = ROOT / "Control_Toolkit" / "Optimizers" / "rpgd_c"
    ext = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}[sys.platform]
    lib_path = c_dir / f"librpgd_cartpole{ext}"
    src = c_dir / "rpgd_cartpole.c"
    header = c_dir / "rpgd_cartpole.h"
    if (not lib_path.exists()) or lib_path.stat().st_mtime < max(src.stat().st_mtime, header.stat().st_mtime):
        optimizer_rpgd_c._build_c_library(c_dir, lib_path.name)
    return configure_c_api(ctypes.CDLL(str(lib_path)))


def tf_optimizer_step(cfg, rt, state, q_init):
    q = tf.Variable(q_init.astype(np.float32))
    m = tf.zeros_like(q)
    v = tf.zeros_like(q)
    for step in range(1, cfg.outer_its + 1):
        with tf.GradientTape() as tape:
            costs = []
            for i in range(cfg.num_rollouts):
                _, cost = tf_rollout_and_cost(cfg, rt, state, q[i])
                costs.append(cost)
            costs = tf.stack(costs)
            loss = tf.reduce_sum(costs)
        grad = tape.gradient(loss, q)
        grad_norm = tf.norm(grad, axis=1, keepdims=True)
        grad = tf.where(grad_norm > cfg.gradmax_clip, grad * (cfg.gradmax_clip / grad_norm), grad)
        m = cfg.adam_beta_1 * m + (1.0 - cfg.adam_beta_1) * grad
        v = cfg.adam_beta_2 * v + (1.0 - cfg.adam_beta_2) * grad * grad
        m_hat = m / (1.0 - cfg.adam_beta_1**step)
        v_hat = v / (1.0 - cfg.adam_beta_2**step)
        q.assign(tf.clip_by_value(q - cfg.learning_rate * m_hat / (tf.sqrt(v_hat) + cfg.adam_epsilon), -1.0, 1.0))
    final_costs = []
    for i in range(cfg.num_rollouts):
        _, cost = tf_rollout_and_cost(cfg, rt, state, q[i])
        final_costs.append(cost)
    final_costs = tf.stack(final_costs).numpy()
    best_idx = int(np.argsort(final_costs)[0])
    return float(q.numpy()[best_idx, 0]), final_costs, np.argsort(final_costs).astype(np.int32)


def tf_rollout_and_cost(cfg, rt, state, q):
    state = tf.convert_to_tensor(state, tf.float32)
    q = tf.convert_to_tensor(q, tf.float32)
    dt = tf.constant(cfg.mpc_timestep / cfg.intermediate_steps, tf.float32)
    target_eq = tf.constant(rt.target_equilibrium if rt.target_equilibrium != 0.0 else 1.0, tf.float32)
    L = tf.constant(rt.L if rt.L > 0.0 else cfg.L, tf.float32)
    m_pole = tf.constant(rt.m_pole if rt.m_pole > 0.0 else cfg.m_pole, tf.float32)
    Lh = 0.5 * L
    kp1 = tf.constant(cfg.k + 1.0, tf.float32)
    total = tf.constant(0.0, tf.float32)
    s = state
    states = [s]
    for h in range(cfg.mpc_horizon):
        position = s[POSITION_IDX]
        angle = s[ANGLE_IDX]
        angleD = s[ANGLED_IDX]
        abs_pos = tf.abs(position)
        T = tf.constant(cfg.track_half_length, tf.float32)
        dd = cfg.dd_quadratic_weight_up * ((position - rt.target_position) / (2.0 * T)) ** 2
        boundary = cfg.permissible_track_fraction * T
        db = tf.cast(abs_pos > boundary, tf.float32) * cfg.db_weight_up * (
            (abs_pos - boundary) / ((1.0 - cfg.permissible_track_fraction) * T)
        ) ** 2
        ep = cfg.ep_weight_up * (1.0 - target_eq * tf.cos(angle)) ** 2
        up_only = (target_eq + 1.0) * 0.5
        kinetic_ref = cfg.vel_penalty_reg * (3.0 * cfg.g / L) * (1.0 - tf.cos(angle))
        ekp = cfg.ekp_weight_up * up_only * tf.abs(angleD**2 - kinetic_ref)
        cc = cfg.cc_weight_up * cfg.R * q[h] ** 2
        total = total + dd + db + ep + ekp + cc
        for _ in range(cfg.intermediate_steps):
            ca = s[ANGLE_COS_IDX]
            sa = s[ANGLE_SIN_IDX]
            angle = s[ANGLE_IDX]
            angleD = s[ANGLED_IDX]
            position = s[POSITION_IDX]
            positionD = s[POSITIOND_IDX]
            u = cfg.u_max * q[h]
            f_fric = -cfg.M_fric * positionD
            t_fric = -cfg.J_fric * angleD
            denom = kp1 * (cfg.m_cart + m_pole) - m_pole * ca * ca
            positionDD = (
                m_pole * cfg.g * sa * ca
                + (t_fric * ca) / Lh
                + kp1 * (-m_pole * Lh * angleD**2 * sa + f_fric + u)
            ) / denom
            angleDD = (cfg.g * sa + positionDD * ca + t_fric / (m_pole * Lh)) / (kp1 * Lh)
            angleD_next = angleD + angleDD * dt
            positionD_next = positionD + positionDD * dt
            angle_next = angle + angleD_next * dt
            position_next = position + positionD_next * dt
            cos_next = tf.cos(angle_next)
            sin_next = tf.sin(angle_next)
            s = tf.stack(
                [tf.atan2(sin_next, cos_next), angleD_next, cos_next, sin_next, position_next, positionD_next]
            )
            states.append(s)
    return tf.stack(states), total / float(cfg.mpc_horizon + 1)


def compare_kernel():
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    q = np.array([0.05 * np.cos(i * 0.2) for i in range(cfg.mpc_horizon)], dtype=np.float32)
    lib = load_c_lib()

    c_state = (ctypes.c_float * 6)(*state)
    c_q = (ctypes.c_float * cfg.mpc_horizon)(*q)
    c_final = (ctypes.c_float * 6)()
    lib.rpgd_debug_rollout_final_state(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, c_final)
    c_cost = lib.rpgd_debug_rollout_cost(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q)
    c_grad = (ctypes.c_float * cfg.mpc_horizon)()
    c_fd = (ctypes.c_float * cfg.mpc_horizon)()
    lib.rpgd_debug_gradient_adjoint(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, c_grad)
    lib.rpgd_debug_gradient_fd(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, c_fd)

    q_tf = tf.Variable(q)
    with tf.GradientTape() as tape:
        states_tf, cost_tf = tf_rollout_and_cost(cfg, rt, state, q_tf)
    grad_tf = tape.gradient(cost_tf, q_tf).numpy()

    final_err = np.max(np.abs(np.array(c_final) - states_tf[-1].numpy()))
    cost_err = abs(c_cost - float(cost_tf.numpy()))
    grad_err = np.max(np.abs(np.array(c_grad) - grad_tf))
    grad_rel = grad_err / max(1.0, float(np.max(np.abs(grad_tf))))
    fd_err = np.max(np.abs(np.array(c_fd) - grad_tf))
    print(f"forward_final_max_abs={final_err:.6g}")
    print(f"cost_abs={cost_err:.6g}")
    print(f"gradient_vs_tf_max_abs={grad_err:.6g} rel={grad_rel:.6g}")
    print(f"finite_difference_vs_tf_max_abs={fd_err:.6g}")
    assert final_err < 2.0e-5
    assert cost_err < 1.0e-4
    assert grad_err < 5.0e-4 or grad_rel < 5.0e-5


def compare_thread_determinism():
    cfg1 = build_config(num_threads=1)
    cfg4 = build_config(num_threads=4)
    rt = RpgdRuntime(0.0, 1.0, cfg1.L, cfg1.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    c_state = (ctypes.c_float * 6)(*state)
    lib = load_c_lib()
    outputs = []
    for cfg in (cfg1, cfg4):
        solver = lib.rpgd_create(ctypes.byref(cfg))
        vals = [lib.rpgd_step(solver, c_state, ctypes.byref(rt)) for _ in range(3)]
        lib.rpgd_destroy(solver)
        outputs.append(vals)
    max_diff = max(abs(a - b) for a, b in zip(outputs[0], outputs[1]))
    print(f"thread_determinism_max_abs={max_diff:.6g}")
    assert max_diff == 0.0


def compare_optimizer_step():
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(1234)
    q_init = rng.uniform(-0.2, 0.2, size=(cfg.num_rollouts, cfg.mpc_horizon)).astype(np.float32)

    tf_u, tf_costs, tf_indices = tf_optimizer_step(cfg, rt, state, q_init)
    lib = load_c_lib()
    solver = lib.rpgd_create(ctypes.byref(cfg))
    lib.rpgd_debug_set_q(solver, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    c_state = (ctypes.c_float * 6)(*state)
    c_u = lib.rpgd_step(solver, c_state, ctypes.byref(rt))
    c_costs = np.empty((cfg.num_rollouts,), dtype=np.float32)
    c_indices = np.empty((cfg.num_rollouts,), dtype=np.int32)
    lib.rpgd_debug_get_costs(solver, c_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.rpgd_debug_get_indices(solver, c_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    lib.rpgd_destroy(solver)

    u_err = abs(c_u - tf_u)
    cost_err = float(np.max(np.abs(c_costs - tf_costs)))
    same_best = int(c_indices[0]) == int(tf_indices[0])
    print(f"optimizer_step_u_abs={u_err:.6g}")
    print(f"optimizer_step_costs_max_abs={cost_err:.6g}")
    print(f"optimizer_step_same_best={same_best}")
    assert u_err < 2.0e-4
    assert cost_err < 1.0e-3
    assert same_best


def compare_reference_library(reference_lib_path):
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    q = np.array([0.05 * np.cos(i * 0.2) for i in range(cfg.mpc_horizon)], dtype=np.float32)
    current = load_c_lib()
    reference = load_c_lib(reference_lib_path)

    c_state = (ctypes.c_float * 6)(*state)
    c_q = (ctypes.c_float * cfg.mpc_horizon)(*q)
    cur_final = (ctypes.c_float * 6)()
    ref_final = (ctypes.c_float * 6)()
    current.rpgd_debug_rollout_final_state(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, cur_final)
    reference.rpgd_debug_rollout_final_state(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, ref_final)
    cur_cost = current.rpgd_debug_rollout_cost(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q)
    ref_cost = reference.rpgd_debug_rollout_cost(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q)
    cur_grad = (ctypes.c_float * cfg.mpc_horizon)()
    ref_grad = (ctypes.c_float * cfg.mpc_horizon)()
    current.rpgd_debug_gradient_adjoint(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, cur_grad)
    reference.rpgd_debug_gradient_adjoint(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, ref_grad)

    final_err = np.max(np.abs(np.array(cur_final) - np.array(ref_final)))
    cost_err = abs(cur_cost - ref_cost)
    grad_err = np.max(np.abs(np.array(cur_grad) - np.array(ref_grad)))

    rng = np.random.default_rng(4321)
    q_init = rng.uniform(-0.2, 0.2, size=(cfg.num_rollouts, cfg.mpc_horizon)).astype(np.float32)
    outputs = []
    costs = []
    indices = []
    for lib in (current, reference):
        solver = lib.rpgd_create(ctypes.byref(cfg))
        lib.rpgd_debug_set_q(solver, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        u = lib.rpgd_step(solver, c_state, ctypes.byref(rt))
        c_costs = np.empty((cfg.num_rollouts,), dtype=np.float32)
        c_indices = np.empty((cfg.num_rollouts,), dtype=np.int32)
        lib.rpgd_debug_get_costs(solver, c_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        lib.rpgd_debug_get_indices(solver, c_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        lib.rpgd_destroy(solver)
        outputs.append(u)
        costs.append(c_costs)
        indices.append(c_indices)

    u_err = abs(outputs[0] - outputs[1])
    step_cost_err = float(np.max(np.abs(costs[0] - costs[1])))
    same_best = int(indices[0][0]) == int(indices[1][0])
    print(f"reference_final_max_abs={final_err:.6g}")
    print(f"reference_cost_abs={cost_err:.6g}")
    print(f"reference_gradient_max_abs={grad_err:.6g}")
    print(f"reference_step_u_abs={u_err:.6g}")
    print(f"reference_step_costs_max_abs={step_cost_err:.6g}")
    print(f"reference_step_same_best={same_best}")
    assert final_err == 0.0
    assert cost_err == 0.0
    assert grad_err == 0.0
    assert u_err == 0.0
    assert step_cost_err == 0.0
    assert same_best


def compare_short_closed_loop():
    state_low = [-np.pi, -np.inf, -1.0, -1.0, -0.22, -np.inf]
    state_high = [-v for v in state_low]
    control_limits = (np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
    attrs = {"target_position": 0.0, "target_equilibrium": 1.0}
    ctrl = controller_mpc(environment_name="CartPole", control_limits=control_limits, initial_environment_attributes=attrs)
    ctrl.configure(optimizer_name="rpgd_c")
    s = create_cartpole_state()
    s[ANGLE_IDX] = 0.2
    s[ANGLED_IDX] = 0.1
    s[ANGLE_COS_IDX] = np.cos(s[ANGLE_IDX])
    s[ANGLE_SIN_IDX] = np.sin(s[ANGLE_IDX])
    controls = []
    for _ in range(10):
        controls.append(float(ctrl.step(s)[0]))
    print(f"closed_loop_smoke_controls_min={min(controls):.6g} max={max(controls):.6g}")
    assert all(np.isfinite(controls))
    assert all(-1.0001 <= u <= 1.0001 for u in controls)


def benchmark():
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    lib = load_c_lib()
    c_state = (ctypes.c_float * 6)(*state)
    c_auto_median = None
    for threads in (1, 2, 4, 0):
        cfg = build_config(num_threads=threads)
        solver = lib.rpgd_create(ctypes.byref(cfg))
        for _ in range(3):
            lib.rpgd_step(solver, c_state, ctypes.byref(rt))
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            lib.rpgd_step(solver, c_state, ctypes.byref(rt))
            times.append((time.perf_counter() - t0) * 1e3)
        lib.rpgd_destroy(solver)
        arr = np.array(times)
        print(
            f"c_threads={threads} min_ms={arr.min():.3f} median_ms={np.median(arr):.3f} p95_ms={np.percentile(arr, 95):.3f}"
        )
        if threads == 0:
            c_auto_median = float(np.median(arr))

    control_limits = (np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
    attrs = {"target_position": 0.0, "target_equilibrium": 1.0}
    ctrl = controller_mpc(environment_name="CartPole", control_limits=control_limits, initial_environment_attributes=attrs)
    ctrl.configure(optimizer_name="rpgd")
    s = create_cartpole_state()
    s[ANGLE_IDX] = state[ANGLE_IDX]
    s[ANGLED_IDX] = state[ANGLED_IDX]
    s[ANGLE_COS_IDX] = state[ANGLE_COS_IDX]
    s[ANGLE_SIN_IDX] = state[ANGLE_SIN_IDX]
    s[POSITION_IDX] = state[POSITION_IDX]
    s[POSITIOND_IDX] = state[POSITIOND_IDX]
    ctrl.step(s)
    py_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        ctrl.step(s)
        py_times.append((time.perf_counter() - t0) * 1e3)
    py_arr = np.array(py_times)
    py_median = float(np.median(py_arr))
    print(
        f"python_rpgd min_ms={py_arr.min():.3f} median_ms={py_median:.3f} p95_ms={np.percentile(py_arr, 95):.3f}"
    )
    if c_auto_median is not None:
        print(f"median_speedup_python_over_c_auto={py_median / c_auto_median:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--reference-lib", type=Path)
    args = parser.parse_args()
    compare_kernel()
    compare_thread_determinism()
    compare_optimizer_step()
    if args.reference_lib is not None:
        compare_reference_library(args.reference_lib)
    compare_short_closed_loop()
    if args.benchmark:
        benchmark()
