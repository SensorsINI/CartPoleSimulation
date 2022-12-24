### Choose whether to run TensorFlow in eager mode (slow, interpreted) or graph mode (fast, compiled)
# Set `USE_TENSORFLOW_EAGER_MODE=False` to:
# 1. decorate functions in optimizers and predictors with `@tf.function`.
# 2. and thereby enable TensorFlow graph mode. This is orders of magnitude faster than the standard eager mode.
# Use USE_TENSORFLOW_EAGER_MODE=True to enable debugging
USE_TENSORFLOW_EAGER_MODE = False


### Choose whether to use TensorFlow Accelerated Linear Algebra (XLA).
# XLA uses machine-specific conversions to speed up the compiled TensorFlow graph.
# Set USE_TENSORFLOW_XLA to True to accelerate the execution (for real-time).
# If `USE_TENSORFLOW_XLA=True`, this adds `jit_compile=True` to the `tf.function` decorator.
# However, XLA ignores random seeds. Set to False for guaranteed reproducibility, such as for simulations.
USE_TENSORFLOW_XLA = True
