import tensorflow as tf

def build_resnet():
    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=base.input, outputs=x)


def build_xception():
    base = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=base.input, outputs=x)


def load_models():
    print("Loading ResNet...")
    resnet = build_resnet()
    resnet.load_weights("saved_model/resnet/weights.h5")

    print("Loading Xception...")
    xception = build_xception()
    xception.load_weights("saved_model/xception/weights.h5")

    return {
        "resnet": resnet,
        "xception": xception
    }