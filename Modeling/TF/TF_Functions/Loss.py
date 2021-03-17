import tensorflow.keras as keras
import tensorflow as tf

# region Define loss and optimizer for training
import numpy as np


def loss_msr_sequence_customizable(wash_out_len, post_wash_out_len, discount_factor=0.9):
    # Calculate discount vector
    discount_vector = np.ones(shape=post_wash_out_len)
    for i in range(post_wash_out_len - 1):
        discount_vector[i + 1] = discount_vector[i] * discount_factor
    discount_vector = tf.convert_to_tensor(discount_vector, dtype=tf.float32)
    print(discount_vector)

    def loss_msr_sequence(y_true, y_predicted):
        losses = keras.losses.MSE(y_true, y_predicted)
        # losses has shape [batch_size, time steps] -> this is the loss for every time step
        losses = losses[:, wash_out_len:]  # This discards losses for timesteps ≤ wash_out_len

        # Get discounted some of losses for a time series
        # Axis (2,1) results in the natural operation of losses * discount_vector
        # loss = keras.layers.Dot(axes=(1, 0))([losses, discount_vector])
        loss = tf.linalg.matvec(losses, discount_vector)

        return loss

    return loss_msr_sequence
