import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from PyTorch_NNs import *
from utils import fast_fillna
from TF_MLP import LiteModel, create_mlp
import joblib


import janestreet
env = janestreet.make_env()
env_iter = env.iter_test()


epochs = 200
batch_size = 4096
hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

# TF Model
tf.keras.backend.clear_session()
clf = create_mlp(
    129, 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )
clf.load_weights('../input/jane-street-with-keras-nn-overfit/model.h5')
tflite_model = LiteModel.from_keras_model(clf)
tf_models = [tflite_model]


# PyTorch Models
embNN_model = Emb_NN_Model()
try:
    embNN_model.load_state_dict(torch.load("../input/embnn5/Jane_EmbNN5_auc_400_400_400.pth"))
except:
    embNN_model.load_state_dict(torch.load("../input/embnn5/Jane_EmbNN5_auc_400_400_400.pth", map_location='cpu'))
embNN_model = embNN_model.eval()

# LightGBM Model
gbm_models = [joblib.load("../input/lgbmfull/lgbfull.pkl")]
fgbm = np.median


for (test_df, pred_df) in tqdm(env_iter):
    if test_df['weight'].values[0] != 0:
        #             x_tt = test_df.values[0][index_features].reshape(1,-1)
        x_tt = test_df.loc[:, features].values
        x_tt[0, :] = fast_fillna(x_tt[0, :], np.zeros(len(features)))

        cross_41_42_43 = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
        cross_1_2 = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
        feature_inp = np.concatenate((
            x_tt,
            np.array(cross_41_42_43).reshape(x_tt.shape[0], 1),
            np.array(cross_1_2).reshape(x_tt.shape[0], 1),
        ), axis=1)

        # torch_pred
        torch_pred = np.zeros((1, len(target_cols)))
        for model in model_list:
            torch_pred += model(
                torch.tensor(feature_inp, dtype=torch.float).to(device)).sigmoid().detach().cpu().numpy() / NFOLDS
        torch_pred = np.median(torch_pred)

        # tf_pred
        tf_pred = np.median(np.mean([tflite_model.predict(x_tt) for model in tf_models], axis=0))

        # torch embedding_NN pred
        x_tt = torch.tensor(x_tt).float().view(-1, 130)
        embnn_p = np.median(torch.sigmoid(embNN_model(None, x_tt)).detach().cpu().numpy().reshape((-1, 5)),
                            axis=1)  # not logits, actually sigmoid probabilities

        gb_pred = fgbm(np.stack([gbmodel.predict(x_tt) for gbmodel in gbm_models]), axis=0).T
        # avg
        pred_pr = torch_pred * 0.308 + tf_pred * 0.5 + embnn_p * 0.112 + gb_pred * 0.08
        #             pred_pr = torch_pred*0.3878 + tf_pred*0.5 + embnn_p*0.1122
        pred_df["action"].values[0] = int(pred_pr >= 0.495)
    else:
        pred_df["action"].values[0] = 0
    env.predict(pred_df)

