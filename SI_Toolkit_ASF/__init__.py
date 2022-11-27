GLOBALLY_DISABLE_COMPILATION = False  # Set to False to use tf.function, set True to use plain python, which will be orders of magnitude slower for integration and other functions
USE_JIT_COMPILATION = True  # XLA ignores random seeds. Set to False for reproducibility
