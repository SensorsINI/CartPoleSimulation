"""Optional angle ADC / position encoder quantization for CartPole simulation.

Mirrors the physical path in ``incoming_data_processor.convert_angle_and_position_skale``:
  angle  = wrap((angle_raw + angle_deviation) * angle_norm)
  position = position_raw * position_norm

Default parameters match the ZYNQ ZEDBOARD + POLOLU rig (``Driver/globals.py``).
"""
from __future__ import annotations

import math

import numpy as np

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.cartpole_parameters import TrackHalfLength
from CartPole.state_utilities import (
    ANGLE_COS_IDX,
    ANGLE_IDX,
    ANGLE_SIN_IDX,
    POSITION_IDX,
)
from others.globals_and_utils import load_config


def _physical_cartpole_params() -> dict[str, float]:
    """Load conversion constants from the physical driver when available."""
    try:
        from globals import (  # noqa: WPS433 — optional physical-cartpole import
            ANGLE_360_DEG_IN_ADC_UNITS,
            ANGLE_DEVIATION,
            ANGLE_HANGING,
            ANGLE_NORMALIZATION_FACTOR,
            POSITION_ENCODER_RANGE,
            POSITION_NORMALIZATION_FACTOR,
            angle_deviation_update,
        )

        angle_hanging = float(ANGLE_HANGING)
        angle_deviation = float(
            ANGLE_DEVIATION.item()
            if hasattr(ANGLE_DEVIATION, "item")
            else angle_deviation_update(angle_hanging)
        )
        return {
            "angle_adc_range": float(ANGLE_360_DEG_IN_ADC_UNITS),
            "angle_deviation": angle_deviation,
            "angle_hanging": angle_hanging,
            "angle_norm": float(ANGLE_NORMALIZATION_FACTOR),
            "position_encoder_range": float(POSITION_ENCODER_RANGE),
            "position_norm": float(POSITION_NORMALIZATION_FACTOR),
        }
    except ImportError:
        angle_adc_range = 4302.0
        angle_hanging = 1068.0
        half_range = angle_adc_range / 2.0
        if angle_hanging < half_range:
            angle_deviation = -angle_hanging - half_range
        else:
            angle_deviation = -angle_hanging + half_range
        return {
            "angle_adc_range": angle_adc_range,
            "angle_deviation": angle_deviation,
            "angle_hanging": angle_hanging,
            "angle_norm": (2.0 * math.pi) / angle_adc_range,
            "position_encoder_range": 4649.0,
            "position_norm": (2.0 * TrackHalfLength) / 4649.0,
        }


def _wrap_angle_raw(angle_raw: float, adc_range: float) -> float:
    half = adc_range / 2.0
    if angle_raw >= half:
        return angle_raw - adc_range
    if angle_raw <= -half:
        return angle_raw + adc_range
    return angle_raw


class SensorQuantizer:
    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config("cartpole_physical_parameters.yml")["cartpole"].get(
                "sensor_quantization", {}
            )

        self.enabled = bool(config.get("enabled", False))
        use_physical = config.get("source", "physical") == "physical"
        params = _physical_cartpole_params() if use_physical else {}

        self.angle_adc_range = float(
            config.get("angle_adc_range", params.get("angle_adc_range", 4302.0))
        )
        self.angle_deviation = float(
            config.get("angle_deviation", params.get("angle_deviation", 0.0))
        )
        self.angle_norm = float(
            config.get("angle_norm", params.get("angle_norm", (2.0 * math.pi) / 4302.0))
        )
        self.position_encoder_range = float(
            config.get(
                "position_encoder_range",
                params.get("position_encoder_range", 4649.0),
            )
        )
        self.position_norm = float(
            config.get("position_norm", params.get("position_norm", (2.0 * TrackHalfLength) / 4649.0))
        )

    def angle_to_raw(self, angle_rad: float) -> float:
        return angle_rad / self.angle_norm - self.angle_deviation

    def raw_to_angle(self, angle_raw: float) -> float:
        wrapped_raw = _wrap_angle_raw(float(angle_raw), self.angle_adc_range)
        return wrap_angle_rad((wrapped_raw + self.angle_deviation) * self.angle_norm)

    def position_to_raw(self, position_m: float) -> float:
        return position_m / self.position_norm

    def raw_to_position(self, position_raw: float) -> float:
        return float(position_raw) * self.position_norm

    def quantize_angle(self, angle_rad: float) -> float:
        raw = self.angle_to_raw(angle_rad)
        return self.raw_to_angle(np.round(raw))

    def quantize_position(self, position_m: float) -> float:
        raw = np.round(self.position_to_raw(position_m))
        return self.raw_to_position(raw)

    def quantize_measurement(self, s, copy: bool = True):
        if not self.enabled:
            return np.copy(s) if copy else s

        out = np.copy(s) if copy else s
        out[ANGLE_IDX] = self.quantize_angle(float(out[ANGLE_IDX]))
        out[ANGLE_COS_IDX] = np.cos(out[ANGLE_IDX])
        out[ANGLE_SIN_IDX] = np.sin(out[ANGLE_IDX])
        out[POSITION_IDX] = self.quantize_position(float(out[POSITION_IDX]))
        return out
