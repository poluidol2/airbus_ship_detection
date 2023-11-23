import tensorflow as tf


def dice_coeff(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_of_squares_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    sum_of_squares_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    dice = (2. * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice


def combined_loss(y_true, y_pred):
    alpha = 1e-3  # Weight for BCE loss
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_coeff(y_true, y_pred)
    combined = alpha * bce_loss + (1. - dice)  # Combine BCE and Dice loss
    return combined
