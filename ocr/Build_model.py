import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dropout, Input, Bidirectional, LSTM, Dense
from tensorflow import keras

def build_model(len_chars, shape_inp_img):
    input_img = Input(shape=(shape_inp_img[0], shape_inp_img[1], shape_inp_img[2]),
                      name="image",
                      dtype="float32")
    base_model = tf.keras.applications.EfficientNetV2L(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=(200, 100, 3),
                                                       input_tensor=input_img,
                                                       include_preprocessing=True)
    x = []
    for layer in base_model.layers:
        if layer.name == "block6a_expand_activation":
            x = Reshape(target_shape=((layer.output_shape[1], int(layer.output_shape[2] * layer.output_shape[3]))))(
                layer.output)
            x = Dropout(0.3)(x)
            break

    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3), merge_mode='ave')(x)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3), merge_mode='ave')(x)
    output = Dense(
        len_chars + 1, activation="softmax", name="dense1"
    )(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output, name="ocr_EfficientNetV2L_v1")

    return model