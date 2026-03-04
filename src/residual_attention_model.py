import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Add, Multiply
from tensorflow.keras.models import Model


def residual_block(x, units):

    shortcut = x

    x = Dense(units, activation="relu")(x)
    x = Dense(units)(x)

    x = Add()([shortcut, x])

    x = tf.keras.layers.Activation("relu")(x)

    return x


def attention_block(x, units):

    attention = Dense(units, activation="softmax")(x)

    x = Multiply()([x, attention])

    return x


def build_model(input_dim, dropout_rate=0.3, learning_rate=0.001):

    inputs = Input(shape=(input_dim,))

    x = Dense(128, activation="relu")(inputs)

    x = residual_block(x, 128)

    x = attention_block(x, 128)

    x = Dense(64, activation="relu")(x)

    x = Dropout(dropout_rate)(x)

    x = Dense(32, activation="relu")(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(name="roc_auc")
        ]
    )

    return model