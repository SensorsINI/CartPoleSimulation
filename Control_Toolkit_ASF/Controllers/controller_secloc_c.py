"""Run the on-chip SecLoc C controller (Firmware/Src/General) on the PC.

Compiles the modular firmware sources (secloc_controller.c + secloc.c + inner
controllers) into a shared library and drives them through ctypes, so the
exact C gate + inner-controller math that runs on the chip can be exercised
from the Python driver.
"""
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


# Stubs for the hardware-facing symbols pulled in by hardware_pid.c; ctypes
# loads with RTLD_NOW, so every referenced symbol must resolve at load time.
_HARDWARE_STUBS_H = """
#ifndef HARDWARE_BRIDGE_H
#define HARDWARE_BRIDGE_H

void enable_irq(void);
void disable_irq(void);
void Message_SendToPC(const unsigned char* data, unsigned int length);

#endif
"""

_HARDWARE_STUBS_C = """
#include <stdbool.h>

void enable_irq(void) {}
void disable_irq(void) {}
void Message_SendToPC(const unsigned char* data, unsigned int length)
{
    (void)data;
    (void)length;
}
unsigned char crc(const unsigned char * message, unsigned int len)
{
    (void)message;
    (void)len;
    return 0;
}
bool crcIsValid(const unsigned char * buff, unsigned int len, unsigned char crcVal)
{
    (void)buff;
    (void)len;
    (void)crcVal;
    return true;
}
void prepare_message_to_PC_config_PID(
    unsigned char * txBuffer,
    float position_KP, float position_KI, float position_KD,
    float angle_KP, float angle_KI, float angle_KD)
{
    (void)txBuffer;
    (void)position_KP; (void)position_KI; (void)position_KD;
    (void)angle_KP; (void)angle_KI; (void)angle_KD;
}
"""


class controller_secloc_c(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        self._source_paths = self._resolve_controller_sources()
        self._library_path = self._build_shared_library()
        self._load_shared_library()

    def _resolve_controller_sources(self):
        firmware_path = Path(self.config_controller["firmware_path"])
        source_names = [self.config_controller["controller_file"]]
        source_names.extend(self.config_controller.get("extra_sources", []))

        source_dir = self._resolve_firmware_dir(firmware_path, source_names[0])
        return [source_dir / name for name in source_names]

    def _resolve_firmware_dir(self, firmware_path, probe_file):
        candidates = []
        if firmware_path.is_absolute():
            candidates.append(firmware_path)
        else:
            cwd = Path.cwd()
            repo_root = Path(__file__).resolve().parents[4]
            cartpole_root = Path(__file__).resolve().parents[2]
            candidates.extend(
                [
                    cwd / firmware_path,
                    cartpole_root / firmware_path,
                    repo_root / firmware_path,
                ]
            )

        for candidate in candidates:
            if (candidate / probe_file).exists():
                return candidate.resolve()

        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"C controller source {probe_file} not found. Tried: {tried}")

    def _build_shared_library(self):
        ops_name = self.config_controller["ops_name"]
        hasher = hashlib.sha256(ops_name.encode("utf-8"))
        for source_path in self._source_paths:
            hasher.update(source_path.read_bytes())
        build_key = hasher.hexdigest()[:16]
        build_dir = Path("/tmp") / "cartpole_c_controllers" / build_key
        build_dir.mkdir(parents=True, exist_ok=True)

        (build_dir / "hardware_bridge.h").write_text(_HARDWARE_STUBS_H, encoding="utf-8")
        stubs_path = build_dir / "hardware_stubs.c"
        stubs_path.write_text(_HARDWARE_STUBS_C, encoding="utf-8")

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

        include_path = self._source_paths[0].parent
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
            str(stubs_path),
            *[str(source_path) for source_path in self._source_paths],
            "-I",
            str(build_dir),  # stubbed hardware_bridge.h shadows none in General
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
            "No C compiler found for secloc-c. Install gcc/clang, set CC, "
            "or add c_compiler to the secloc-c controller config."
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
        if name == "target_equilibrium":
            return float(
                np.asarray(getattr(self.variable_parameters, "target_equilibrium", 1.0)).item()
            )
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
