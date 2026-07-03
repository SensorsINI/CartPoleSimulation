import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from SI_Toolkit.computation_library import NumpyLibrary, TensorType

from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from Control_Toolkit.Controllers import template_controller


class controller_secloc_lqr_c(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        self._source_path = self._resolve_controller_source()
        self._library_path = self._build_shared_library()
        self._load_shared_library()

    def _resolve_controller_source(self):
        firmware_path = Path(self.config_controller["firmware_path"])
        controller_file = self.config_controller["controller_file"]

        candidates = []
        if firmware_path.is_absolute():
            candidates.append(firmware_path / controller_file)
        else:
            cwd = Path.cwd()
            repo_root = Path(__file__).resolve().parents[4]
            cartpole_root = Path(__file__).resolve().parents[2]
            candidates.extend(
                [
                    cwd / firmware_path / controller_file,
                    cartpole_root / firmware_path / controller_file,
                    repo_root / firmware_path / controller_file,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"C controller source {controller_file} not found. Tried: {tried}")

    def _build_shared_library(self):
        ops_name = self.config_controller["ops_name"]
        source_bytes = self._source_path.read_bytes()
        build_key = hashlib.sha256(source_bytes + ops_name.encode("utf-8")).hexdigest()[:16]
        build_dir = Path("/tmp") / "cartpole_c_controllers" / build_key
        build_dir.mkdir(parents=True, exist_ok=True)

        wrapper_path = build_dir / "controller_wrapper.c"
        library_path = build_dir / "controller_wrapper.so"
        wrapper_path.write_text(
            f"""
#include "controller_api.h"

extern const ControllerOps {ops_name};

const ControllerSpec* controller_spec(void)
{{
    return {ops_name}.spec();
}}

int controller_n_inputs(void)
{{
    return controller_spec()->n_inputs;
}}

const char* controller_input_name(int index)
{{
    const ControllerSpec* spec = controller_spec();
    if (index < 0 || index >= spec->n_inputs) return "";
    return spec->names[index];
}}

void controller_init(void)
{{
    if ({ops_name}.init) {ops_name}.init();
}}

float controller_step(const float* inputs)
{{
    float outputs[1] = {{0.0f}};
    {ops_name}.evaluate(inputs, outputs);
    return outputs[0];
}}
""",
            encoding="utf-8",
        )

        include_path = self._source_path.parent
        compiler = self._select_c_compiler()
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-std=c99",
            "-O2",
            "-o",
            str(library_path),
            str(wrapper_path),
            str(self._source_path),
            "-I",
            str(include_path),
            "-lm",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return library_path

    def _select_c_compiler(self):
        configured_compiler = self.config_controller.get("c_compiler")
        candidates = [configured_compiler, os.environ.get("CC"), "gcc", "cc", "clang"]
        for candidate in candidates:
            if candidate and shutil.which(candidate):
                return candidate

        raise RuntimeError(
            "No C compiler found for secloc-lqr-c. Install gcc/clang, set CC, "
            "or add c_compiler to the secloc-lqr-c controller config."
        )

    def _load_shared_library(self):
        self._lib = ctypes.CDLL(str(self._library_path))
        self._lib.controller_init.argtypes = []
        self._lib.controller_init.restype = None
        self._lib.controller_n_inputs.argtypes = []
        self._lib.controller_n_inputs.restype = ctypes.c_int
        self._lib.controller_input_name.argtypes = [ctypes.c_int]
        self._lib.controller_input_name.restype = ctypes.c_char_p
        self._lib.controller_step.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.controller_step.restype = ctypes.c_float

        self._lib.controller_init()
        self.input_names = [
            self._lib.controller_input_name(i).decode("ascii")
            for i in range(self._lib.controller_n_inputs())
        ]

    def _value_for_input(self, name, s, time):
        if name == "angle":
            return float(s[ANGLE_IDX])
        if name == "angleD":
            return float(s[ANGLED_IDX])
        if name == "position":
            return float(s[POSITION_IDX])
        if name == "positionD":
            return float(s[POSITIOND_IDX])
        if name == "target_position":
            return float(np.asarray(self.variable_parameters.target_position).item())
        if name == "time":
            return 0.0 if time is None else float(time)
        raise ValueError(f"C controller requested unknown input {name!r}")

    def controller_reset(self):
        if hasattr(self, "_lib"):
            self._lib.controller_init()

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType] | None" = None):
        if updated_attributes is None:
            updated_attributes = {}
        self.update_attributes(updated_attributes)

        values = [self._value_for_input(name, s, time) for name in self.input_names]
        inputs = (ctypes.c_float * len(values))(*values)
        return np.float32(self._lib.controller_step(inputs))
