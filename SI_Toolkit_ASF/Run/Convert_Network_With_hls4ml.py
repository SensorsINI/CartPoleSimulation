from SI_Toolkit.HLS4ML.convert_with_hls4ml import convert_with_hls4ml

from post_hls4ml_marshal_config import generate_marshal_config_after_hls4ml

convert_with_hls4ml()

try:
    generate_marshal_config_after_hls4ml()
except Exception as exc:  # noqa: BLE001 - conversion already succeeded
    print(f"Warning: nn_marshal_config.h was not generated: {exc}")
