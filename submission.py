import pandas as pd
import numpy as np
import tensorflow as tf
from TF_MLP import LiteModel, create_mlp
import janestreet
env = janestreet.make_env()
env_iter = env.iter_test()


epochs = 200
batch_size = 4096
hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

tf.keras.backend.clear_session()
clf = create_mlp(
    129, 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )
clf.load_weights('../input/jane-street-with-keras-nn-overfit/model.h5')
tflite_model = LiteModel.from_keras_model(clf)
tf_models = [tflite_model]


