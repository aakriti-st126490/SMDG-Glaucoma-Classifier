import tensorflow as tf
import numpy as np

def get_last_conv_layer(model_name):
    if model_name == "resnet":
        return "conv5_block3_out"
    elif model_name == "xception":
        return "block14_sepconv2_act"


def generate_gradcam(model, img, model_name):
    last_conv_layer_name = get_last_conv_layer(model_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()