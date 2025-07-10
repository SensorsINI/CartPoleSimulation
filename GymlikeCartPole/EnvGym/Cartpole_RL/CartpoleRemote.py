#!/usr/bin/env python3
"""
Cartpole_Remote.py

A CartPoleSimulatorBase back-end that proxies every (state ↔ action)
exchange to the real-world cart-pole over ZeroMQ.

Protocol
--------
    • The hardware (client) opens a DEALER/REQ socket and **sends**
      one JSON frame:

           {"rid": n, "state": [...], "time": t, "updated_attributes": {...}}

      then blocks, waiting for a reply.

    • This class (server, inside the Gym env) replies with

           {"rid": n, "Q": [u]}

      where `u` is the control command in -1…1.

    • Immediately after receiving the reply the hardware executes the
      command, measures the next state, and starts the next cycle.

Timing
------
The hardware is paused between `reset()` and the first `env.step(a0)`
call – exactly like a simulator.

Author: 2025-07-09
"""
from __future__ import annotations
import json
import zmq
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from SI_Toolkit.computation_library import NumpyLibrary
from others.globals_and_utils import load_config
from CartPole.cartpole_equations import CartPoleEquations
from GymlikeCartPole.EnvGym.Cartpole_RL._cartpole_rl_template import (
    CartPoleSimulatorBase,
)


class Cartpole_Remote(CartPoleSimulatorBase):
    """
    Thin proxy between Gym and the real hardware.
    All heavy lifting is done in :pymeth:`next_state`.
    """

    # ------------------------- initialisation -------------------------
    def __init__(
        self,
        *,
        endpoint: str = "tcp://*:5555",
    ):
        # --- static params (for reward / termination geometry) --------

        config = load_config("cartpole_physical_parameters.yml")["cartpole"]
        self.cpe = CartPoleEquations(
            lib=NumpyLibrary(),
            second_derivatives_mode=config["second_derivatives_mode"],
            second_derivatives_neural_model_path=config["second_derivatives_neural_model_path"]
            )

        self.x_limit = self.cpe.params.TrackHalfLength
        self.pole_length = self.cpe.params.L

        # observation / action spaces identical to the custom sim
        high = np.array(
            [
                np.pi,                  # θ
                np.inf,                 # θ̇
                1.0, 1.0,               # sin θ, cos θ
                self.x_limit * 2,       # x
                np.inf,                 # ẋ
            ],
            dtype=np.float32,
        )
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # --- ZeroMQ ---------------------------------------------------
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(endpoint)

        # will be filled by reset()
        self._pending_identity: Optional[bytes] = None
        self._pending_rid:      Optional[int]        = None
        self._pending_state:    Optional[np.ndarray] = None

    # ------------------------- helper I/O ----------------------------
    def _recv_state(self) -> Tuple[bytes, int, np.ndarray]:
        """
        Blocks until a state frame arrives.
        Returns (identity, rid, state ∈ ℝ⁶).
        """
        parts = self._sock.recv_multipart()
        if len(parts) == 2:
            identity, payload = parts
        elif len(parts) == 3 and parts[1] == b"":
            identity, _empty, payload = parts
        else:
            raise RuntimeError("Unexpected ZeroMQ frame format")

        data  = json.loads(payload.decode("utf-8"))
        rid   = data["rid"]                                  # «―――»
        state = np.asarray(data["state"], dtype=np.float32)
        return identity, rid, state

    def _send_action(self, identity: bytes, rid: int, action: np.ndarray) -> None:
        """
        Reply with the control command encoded as ``{"rid": rid, "Q": [u]}``.
        """
        u      = float(np.squeeze(action))          # ensure JSON-serialisable
        frame  = json.dumps({"rid": rid, "Q": [u]}).encode()
        self._sock.send_multipart([identity, frame])

    # --------------------- CartPoleSimulatorBase API -----------------
    def reset(self) -> None:
        """
        Wait for the very first hardware state.
        **No reply is sent yet** – the first action will be dispatched
        when the agent calls ``env.step(a0)``.
        """
        self._pending_identity, self._pending_rid, self._pending_state = self._recv_state()

    # NB: current_state is ignored – the hardware *is* the ground truth
    def next_state(self, _current_state: np.ndarray,
                   action: np.ndarray) -> np.ndarray:
        """
        (1) send ``action`` to the hardware,
        (2) receive and return the next measured state.
        """
        if self._pending_identity is None:
            raise RuntimeError("reset() must be called before next_state()")

        # 1) close the request–reply cycle
        self._send_action(self._pending_identity, self._pending_rid, action)              # «―――»

        # 2) immediately wait for the next measurement
        self._pending_identity, self._pending_rid, self._pending_state = self._recv_state()
        return self._pending_state.astype(np.float32)

    # handy accessor used by CartPoleEnv.reset()
    def get_initial_state(self) -> np.ndarray:
        if self._pending_state is None:
            raise RuntimeError("reset() not yet called")
        return self._pending_state

    # ------------------------- housekeeping --------------------------
    def __del__(self):
        try:
            self._sock.close(0)
            self._ctx.term()
        except Exception:
            pass
