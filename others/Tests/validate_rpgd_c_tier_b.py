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
    lib.rpgd_debug_rollout_final_state.restype = None
    lib.rpgd_debug_gradient_adjoint.argtypes = [
        ctypes.POINTER(RpgdConfig),
        ctypes.POINTER(RpgdRuntime),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rpgd_debug_gradient_adjoint.restype = None
    lib.rpgd_debug_gradient_fd.argtypes = lib.rpgd_debug_gradient_adjoint.argtypes
    lib.rpgd_debug_gradient_fd.restype = None
    lib.rpgd_create.argtypes = [ctypes.POINTER(RpgdConfig)]
    lib.rpgd_create.restype = ctypes.c_void_p
    lib.rpgd_destroy.argtypes = [ctypes.c_void_p]
    lib.rpgd_destroy.restype = None
    lib.rpgd_debug_set_q.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rpgd_debug_set_q.restype = None
    lib.rpgd_debug_get_adam.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.rpgd_debug_get_adam.restype = None
    lib.rpgd_debug_get_costs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rpgd_debug_get_costs.restype = None
    lib.rpgd_debug_get_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.rpgd_debug_get_indices.restype = None
    lib.rpgd_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(RpgdRuntime),
    ]
    lib.rpgd_step.restype = ctypes.c_float
    if hasattr(lib, "rpgd_is_baremetal"):
        lib.rpgd_is_baremetal.argtypes = []
        lib.rpgd_is_baremetal.restype = ctypes.c_int
    if hasattr(lib, "rpgd_get_workspace_bytes"):
        lib.rpgd_get_workspace_bytes.argtypes = [ctypes.c_void_p]
        lib.rpgd_get_workspace_bytes.restype = ctypes.c_size_t
    if hasattr(lib, "rpgd_get_last_status"):
        lib.rpgd_get_last_status.argtypes = [ctypes.c_void_p]
        lib.rpgd_get_last_status.restype = ctypes.c_int
    if hasattr(lib, "rpgd_get_static_workspace_bytes"):
        lib.rpgd_get_static_workspace_bytes.argtypes = []
        lib.rpgd_get_static_workspace_bytes.restype = ctypes.c_size_t
    if hasattr(lib, "rpgd_validate_config"):
        lib.rpgd_validate_config.argtypes = [ctypes.POINTER(RpgdConfig)]
        lib.rpgd_validate_config.restype = ctypes.c_int
    if hasattr(lib, "rpgd_get_worker_scratch_bytes"):
        lib.rpgd_get_worker_scratch_bytes.argtypes = []
        lib.rpgd_get_worker_scratch_bytes.restype = ctypes.c_size_t
    if hasattr(lib, "rpgd_step_prepare"):
        lib.rpgd_step_prepare.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(RpgdRuntime),
            ctypes.c_void_p,
        ]
        lib.rpgd_step_prepare.restype = ctypes.c_int
        lib.rpgd_step_optimize_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.rpgd_step_optimize_range.restype = ctypes.c_int
        lib.rpgd_step_finalize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.rpgd_step_finalize.restype = ctypes.c_float
        lib.rpgd_step_abort.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rpgd_step_abort.restype = None
    if hasattr(lib, "rpgd_is_busy"):
        lib.rpgd_is_busy.argtypes = [ctypes.c_void_p]
        lib.rpgd_is_busy.restype = ctypes.c_int
    if hasattr(lib, "rpgd_get_resample_phase"):
        lib.rpgd_get_resample_phase.argtypes = [ctypes.c_void_p]
        lib.rpgd_get_resample_phase.restype = ctypes.c_int
    if hasattr(lib, "rpgd_debug_get_q"):
        lib.rpgd_debug_get_q.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.rpgd_debug_get_q.restype = None
    return lib


def _c_dir():
    return ROOT / "Control_Toolkit" / "Optimizers" / "rpgd_c"


def _c_source_mtime(c_dir):
    sources = [
        c_dir / "rpgd_cartpole.c",
        c_dir / "cartpole_model.c",
        c_dir / "cartpole_cost.c",
        c_dir / "rpgd_cartpole.h",
        c_dir / "cartpole_model.h",
        c_dir / "cartpole_cost.h",
        c_dir / "rpgd_platform.h",
        c_dir / "rpgd_worker.h",
    ]
    return max(path.stat().st_mtime for path in sources)


def load_c_lib(path=None):
    if path is not None:
        return configure_c_api(ctypes.CDLL(str(path)))
    c_dir = _c_dir()
    ext = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}[sys.platform]
    lib_path = c_dir / f"librpgd_cartpole{ext}"
    if (not lib_path.exists()) or lib_path.stat().st_mtime < _c_source_mtime(c_dir):
        optimizer_rpgd_c._build_c_library(c_dir, lib_path.name)
    return configure_c_api(ctypes.CDLL(str(lib_path)))


def load_baremetal_lib():
    c_dir = _c_dir()
    ext = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}[sys.platform]
    lib_path = c_dir / f"librpgd_cartpole_baremetal{ext}"
    if (not lib_path.exists()) or lib_path.stat().st_mtime < _c_source_mtime(c_dir):
        optimizer_rpgd_c._build_c_library(
            c_dir,
            lib_path.name,
            extra_cflags=["-DRPGD_BAREMETAL"],
            allow_openmp=False,
        )
    lib = configure_c_api(ctypes.CDLL(str(lib_path)))
    assert lib.rpgd_is_baremetal() == 1
    return lib


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


def compare_baremetal_library():
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    q = np.array([0.05 * np.cos(i * 0.2) for i in range(cfg.mpc_horizon)], dtype=np.float32)
    current = load_c_lib()
    baremetal = load_baremetal_lib()
    assert current.rpgd_is_baremetal() == 0
    assert baremetal.rpgd_is_baremetal() == 1

    c_state = (ctypes.c_float * 6)(*state)
    c_q = (ctypes.c_float * cfg.mpc_horizon)(*q)
    cur_final = (ctypes.c_float * 6)()
    bm_final = (ctypes.c_float * 6)()
    current.rpgd_debug_rollout_final_state(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, cur_final)
    baremetal.rpgd_debug_rollout_final_state(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, bm_final)
    cur_cost = current.rpgd_debug_rollout_cost(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q)
    bm_cost = baremetal.rpgd_debug_rollout_cost(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q)
    cur_grad = (ctypes.c_float * cfg.mpc_horizon)()
    bm_grad = (ctypes.c_float * cfg.mpc_horizon)()
    current.rpgd_debug_gradient_adjoint(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, cur_grad)
    baremetal.rpgd_debug_gradient_adjoint(ctypes.byref(cfg), ctypes.byref(rt), c_state, c_q, bm_grad)

    final_err = np.max(np.abs(np.array(cur_final) - np.array(bm_final)))
    cost_err = abs(cur_cost - bm_cost)
    grad_err = np.max(np.abs(np.array(cur_grad) - np.array(bm_grad)))

    rng = np.random.default_rng(4321)
    q_init = rng.uniform(-0.2, 0.2, size=(cfg.num_rollouts, cfg.mpc_horizon)).astype(np.float32)
    outputs = []
    costs = []
    indices = []
    sequences = []
    for lib in (current, baremetal):
        solver = lib.rpgd_create(ctypes.byref(cfg))
        lib.rpgd_debug_set_q(solver, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        seq = [lib.rpgd_step(solver, c_state, ctypes.byref(rt)) for _ in range(3)]
        c_costs = np.empty((cfg.num_rollouts,), dtype=np.float32)
        c_indices = np.empty((cfg.num_rollouts,), dtype=np.int32)
        lib.rpgd_debug_get_costs(solver, c_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        lib.rpgd_debug_get_indices(solver, c_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        lib.rpgd_destroy(solver)
        outputs.append(seq[0])
        sequences.append(seq)
        costs.append(c_costs)
        indices.append(c_indices)

    rng_seq = []
    for lib in (current, baremetal):
        solver = lib.rpgd_create(ctypes.byref(cfg))
        rng_seq.append([lib.rpgd_step(solver, c_state, ctypes.byref(rt)) for _ in range(3)])
        lib.rpgd_destroy(solver)

    u_err = abs(outputs[0] - outputs[1])
    step_cost_err = float(np.max(np.abs(costs[0] - costs[1])))
    seq_err = max(abs(a - b) for a, b in zip(sequences[0], sequences[1]))
    rng_err = max(abs(a - b) for a, b in zip(rng_seq[0], rng_seq[1]))
    same_best = int(indices[0][0]) == int(indices[1][0])
    print(f"baremetal_final_max_abs={final_err:.6g}")
    print(f"baremetal_cost_abs={cost_err:.6g}")
    print(f"baremetal_gradient_max_abs={grad_err:.6g}")
    print(f"baremetal_step_u_abs={u_err:.6g}")
    print(f"baremetal_step_seq_max_abs={seq_err:.6g}")
    print(f"baremetal_rng_seq_max_abs={rng_err:.6g}")
    print(f"baremetal_step_costs_max_abs={step_cost_err:.6g}")
    print(f"baremetal_step_same_best={same_best}")
    assert final_err == 0.0
    assert cost_err == 0.0
    assert grad_err == 0.0
    assert u_err == 0.0
    assert seq_err == 0.0
    assert rng_err == 0.0
    assert step_cost_err == 0.0
    assert same_best


def compare_safety_guards():
    current = load_c_lib()
    baremetal = load_baremetal_lib()
    cfg = build_config(num_threads=1)

    invalid_cfg = build_config(num_threads=1)
    invalid_cfg.adam_beta_2 = np.nan
    assert current.rpgd_validate_config(ctypes.byref(invalid_cfg)) != 0
    assert not current.rpgd_create(ctypes.byref(invalid_cfg))

    solver = baremetal.rpgd_create(ctypes.byref(cfg))
    assert solver
    assert not baremetal.rpgd_create(ctypes.byref(cfg))
    assert baremetal.rpgd_get_workspace_bytes(solver) == baremetal.rpgd_get_static_workspace_bytes()

    state = np.array(
        [np.nan, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0],
        dtype=np.float32,
    )
    runtime = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    output = baremetal.rpgd_step(
        solver,
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(runtime),
    )
    assert output == 0.0
    assert baremetal.rpgd_get_last_status(solver) != 0
    baremetal.rpgd_destroy(solver)

    warm_cfg = build_config(num_threads=1)
    warm_cfg.outer_its = 2
    warm_cfg.warmup = 1
    warm_cfg.warmup_iterations = 7
    warm_solver = baremetal.rpgd_create(ctypes.byref(warm_cfg))
    assert warm_solver
    finite_state = np.array(
        [0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0],
        dtype=np.float32,
    )
    baremetal.rpgd_step(
        warm_solver,
        finite_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(runtime),
    )
    adam_step = ctypes.c_int()
    baremetal.rpgd_debug_get_adam(warm_solver, None, None, ctypes.byref(adam_step))
    assert adam_step.value == 7
    baremetal.rpgd_step(
        warm_solver,
        finite_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(runtime),
    )
    baremetal.rpgd_debug_get_adam(warm_solver, None, None, ctypes.byref(adam_step))
    assert adam_step.value == 9
    baremetal.rpgd_destroy(warm_solver)
    print("safety_guards=PASS")


class RpgdStepPlan(ctypes.Structure):
    _fields_ = [
        ("state6", ctypes.c_float * 6),
        ("runtime", RpgdRuntime),
        ("active_iterations", ctypes.c_int),
        ("prepared", ctypes.c_int),
        ("range_error", ctypes.c_int),
    ]


def _snapshot(lib, solver, cfg):
    n = cfg.num_rollouts * cfg.mpc_horizon
    q = np.empty(n, dtype=np.float32)
    m = np.empty(n, dtype=np.float32)
    v = np.empty(n, dtype=np.float32)
    costs = np.empty(cfg.num_rollouts, dtype=np.float32)
    indices = np.empty(cfg.num_rollouts, dtype=np.int32)
    step = ctypes.c_int()
    lib.rpgd_debug_get_q(solver, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.rpgd_debug_get_adam(
        solver,
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(step),
    )
    lib.rpgd_debug_get_costs(solver, costs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    lib.rpgd_debug_get_indices(solver, indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    return {
        "q": q.copy(),
        "m": m.copy(),
        "v": v.copy(),
        "costs": costs.copy(),
        "indices": indices.copy(),
        "adam_step": step.value,
        "resample": lib.rpgd_get_resample_phase(solver),
        "status": lib.rpgd_get_last_status(solver),
    }


def _split_step(lib, solver, state, rt, ranges):
    plan = RpgdStepPlan()
    scratch_size = lib.rpgd_get_worker_scratch_bytes()
    rc = lib.rpgd_step_prepare(solver, state, ctypes.byref(rt), ctypes.byref(plan))
    if rc != 0:
        return 0.0, rc
    for first, last in ranges:
        scratch = ctypes.create_string_buffer(scratch_size)
        rc = lib.rpgd_step_optimize_range(
            solver, ctypes.byref(plan), first, last, scratch
        )
        if rc != 0:
            lib.rpgd_step_abort(solver, rc)
            return 0.0, rc
    u = lib.rpgd_step_finalize(solver, ctypes.byref(plan))
    return u, lib.rpgd_get_last_status(solver)


def compare_split_phases():
    cfg = build_config(num_threads=1)
    rt = RpgdRuntime(0.0, 1.0, cfg.L, cfg.m_pole)
    state = np.array([0.2, 0.1, np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32)
    c_state = (ctypes.c_float * 6)(*state)
    rng = np.random.default_rng(99)
    q_init = rng.uniform(-0.2, 0.2, size=(cfg.num_rollouts, cfg.mpc_horizon)).astype(np.float32)
    n = cfg.num_rollouts
    partitions = {
        "one_range": [(0, n)],
        "8_8": [(0, 8), (8, n)],
        "7_9": [(0, 7), (7, n)],
        "9_7": [(0, 9), (9, n)],
        "four": [(0, 4), (4, 8), (8, 12), (12, n)],
    }

    lib = load_c_lib()
    bm = load_baremetal_lib()
    for name, lib_i in (("host", lib), ("baremetal", bm)):
        mono = lib_i.rpgd_create(ctypes.byref(cfg))
        lib_i.rpgd_debug_set_q(mono, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        mono_seq = []
        mono_snaps = []
        for _ in range(12):
            mono_seq.append(lib_i.rpgd_step(mono, c_state, ctypes.byref(rt)))
            mono_snaps.append(_snapshot(lib_i, mono, cfg))
        lib_i.rpgd_destroy(mono)

        for part_name, ranges in partitions.items():
            solver = lib_i.rpgd_create(ctypes.byref(cfg))
            lib_i.rpgd_debug_set_q(solver, q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            for i, expected in enumerate(mono_seq):
                u, status = _split_step(lib_i, solver, c_state, rt, ranges)
                snap = _snapshot(lib_i, solver, cfg)
                assert status == 0
                assert u == expected, f"{name} {part_name} step {i}: {u} vs {expected}"
                for key in ("q", "m", "v", "costs", "indices"):
                    assert np.array_equal(snap[key], mono_snaps[i][key]), f"{name} {part_name} {key}"
                assert snap["adam_step"] == mono_snaps[i]["adam_step"]
                assert snap["resample"] == mono_snaps[i]["resample"]
            lib_i.rpgd_destroy(solver)
        print(f"split_phases_{name}=PASS")

    solver = bm.rpgd_create(ctypes.byref(cfg))
    plan = RpgdStepPlan()
    scratch = ctypes.create_string_buffer(bm.rpgd_get_worker_scratch_bytes())
    rc = bm.rpgd_step_prepare(solver, c_state, ctypes.byref(rt), ctypes.byref(plan))
    assert rc == 0
    assert bm.rpgd_is_busy(solver) == 1
    rc_busy = bm.rpgd_step_prepare(solver, c_state, ctypes.byref(rt), ctypes.byref(plan))
    assert rc_busy != 0
    bm.rpgd_step_abort(solver, 0)
    assert bm.rpgd_is_busy(solver) == 0

    phase_before = bm.rpgd_get_resample_phase(solver)
    step = ctypes.c_int()
    bm.rpgd_debug_get_adam(solver, None, None, ctypes.byref(step))
    adam_before = step.value
    rc = bm.rpgd_step_prepare(solver, c_state, ctypes.byref(rt), ctypes.byref(plan))
    assert rc == 0
    bm.rpgd_step_abort(solver, -7)
    assert bm.rpgd_is_busy(solver) == 0
    assert bm.rpgd_get_resample_phase(solver) == phase_before
    bm.rpgd_debug_get_adam(solver, None, None, ctypes.byref(step))
    assert step.value == adam_before

    rc = bm.rpgd_step_prepare(solver, c_state, ctypes.byref(rt), ctypes.byref(plan))
    assert rc == 0
    assert bm.rpgd_step_optimize_range(solver, ctypes.byref(plan), 0, 99, scratch) != 0
    assert plan.range_error != 0
    u_bad = bm.rpgd_step_finalize(solver, ctypes.byref(plan))
    assert u_bad == 0.0
    assert bm.rpgd_is_busy(solver) == 0
    bm.rpgd_debug_get_adam(solver, None, None, ctypes.byref(step))
    assert step.value == adam_before

    rc = bm.rpgd_step_prepare(solver, c_state, ctypes.byref(rt), ctypes.byref(plan))
    assert rc == 0
    assert bm.rpgd_step_optimize_range(solver, ctypes.byref(plan), 0, n, scratch) == 0
    u = bm.rpgd_step_finalize(solver, ctypes.byref(plan))
    assert np.isfinite(u)
    u2 = bm.rpgd_step_finalize(solver, ctypes.byref(plan))
    assert u2 == 0.0
    bm.rpgd_destroy(solver)
    print("split_failure_injection=PASS")


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
    compare_baremetal_library()
    compare_safety_guards()
    compare_split_phases()
    if args.reference_lib is not None:
        compare_reference_library(args.reference_lib)
    compare_short_closed_loop()
    if args.benchmark:
        benchmark()
