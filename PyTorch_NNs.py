import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

DATA_PATH = '.'
NFOLDS = 5
TRAIN = False
CACHE_PATH = 'archive/'
N_FEAT_TAGS = 29  # No of tags in features.csv
N_FEATURES = 130
THREE_HIDDEN_LAYERS = [400, 400, 400]


feat_cols = ['feature_' + str(i) for i in range(130)]
target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']
all_feat_cols = [col for col in feat_cols]
all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        dropout_rate = 0.2
        hidden_size = 256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size + len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size + hidden_size, len(target_cols))

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)
        x = self.dense5(x)

        return x


if True:
    device = torch.device("cpu")

    model_list = []
    tmp = np.zeros(len(feat_cols))
    for _fold in range(NFOLDS):
        torch.cuda.empty_cache()
        model = Model()
        model.to(device)
        model_weights = CACHE_PATH + 'online_model' + str(_fold) + '.pth'
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
        model.eval()
        model_list.append(model)


class FFN(nn.Module):

    def __init__(self, inputCount=130, outputCount=5, hiddenLayerCounts=[150, 150, 150],
                 drop_prob=0.2, nonlin=nn.SiLU(), isOpAct=False):

        super(FFN, self).__init__()

        self.nonlin = nonlin
        self.dropout = nn.Dropout(drop_prob)
        self.batchnorm0 = nn.BatchNorm1d(inputCount)
        self.dense1 = nn.Linear(inputCount, hiddenLayerCounts[0])
        self.batchnorm1 = nn.BatchNorm1d(hiddenLayerCounts[0])
        self.dense2 = nn.Linear(hiddenLayerCounts[0], hiddenLayerCounts[1])
        self.batchnorm2 = nn.BatchNorm1d(hiddenLayerCounts[1])
        self.dense3 = nn.Linear(hiddenLayerCounts[1], hiddenLayerCounts[2])
        self.batchnorm3 = nn.BatchNorm1d(hiddenLayerCounts[2])
        self.outDense = None
        if outputCount > 0:
            self.outDense = nn.Linear(hiddenLayerCounts[-1], outputCount)
        self.outActivtn = None
        if isOpAct:
            if outputCount == 1 or outputCount == 2:
                self.outActivtn = nn.Sigmoid()
            elif outputCount > 0:
                self.outActivtn = nn.Softmax(dim=-1)
        return

    def forward(self, X):

        # X = self.dropout (self.batchnorm0 (X))
        X = self.batchnorm0(X)
        X = self.dropout(self.nonlin(self.batchnorm1(self.dense1(X))))
        X = self.dropout(self.nonlin(self.batchnorm2(self.dense2(X))))
        X = self.dropout(self.nonlin(self.batchnorm3(self.dense3(X))))
        if self.outDense:
            X = self.outDense(X)
        if self.outActivtn:
            X = self.outActivtn(X)
        return X


class Emb_NN_Model(nn.Module):

    def __init__(self, three_hidden_layers=THREE_HIDDEN_LAYERS, embed_dim=(N_FEAT_TAGS),
                 csv_file='features.csv'):
        super(Emb_NN_Model, self).__init__()
        global N_FEAT_TAGS
        N_FEAT_TAGS = 29

        # store the features to tags mapping as a datframe tdf, feature_i mapping is in tdf[i, :]
        dtype = {'tag_0': 'int8'}
        for i in range(1, 29):
            k = 'tag_' + str(i)
            dtype[k] = 'int8'
        t_df = pd.read_csv(csv_file, usecols=range(1, N_FEAT_TAGS + 1), dtype=dtype)
        t_df['tag_29'] = np.array([1] + ([0] * (t_df.shape[0] - 1))).astype('int8')
        self.features_tag_matrix = torch.tensor(t_df.to_numpy())
        N_FEAT_TAGS += 1


        # embeddings for the tags. Each feature is taken a an embedding which is an avg. of its' tag embeddings
        self.embed_dim = embed_dim
        self.tag_embedding = nn.Embedding(N_FEAT_TAGS + 1,
                                          embed_dim)  # create a special tag if not known tag for any feature
        self.tag_weights = nn.Linear(N_FEAT_TAGS, 1)

        drop_prob = 0.5
        self.ffn = FFN(inputCount=(130 + embed_dim), outputCount=0,
                       hiddenLayerCounts=[(three_hidden_layers[0] + embed_dim), (three_hidden_layers[1] + embed_dim),
                                          (three_hidden_layers[2] + embed_dim)], drop_prob=drop_prob)
        self.outDense = nn.Linear(three_hidden_layers[2] + embed_dim, 5)
        return

    def features2emb(self):
        """
        idx : int feature index 0 to N_FEATURES-1 (129)
        """

        all_tag_idxs = torch.LongTensor(np.arange(N_FEAT_TAGS))  # .to (DEVICE)              # (29,)
        tag_bools = self.features_tag_matrix  # (130, 29)
        f_emb = self.tag_embedding(all_tag_idxs).repeat(130, 1,
                                                        1)
        f_emb = f_emb * tag_bools[:, :,None]
        # Take avg. of all the present tag's embeddings to get the embedding for a feature
        s = torch.sum(tag_bools, dim=1)  # (130,)

        f_emb = torch.sum(f_emb, dim=-2) / s[:, None]  # (130, 7)
        return f_emb

    def forward(self, cat_featrs, features):
        """
        when you call `model (x ,y, z, ...)` then this method is invoked
        """

        cat_featrs = None
        features = features.view(-1, N_FEATURES)
        f_emb = self.features2emb()
        features_2 = torch.matmul(features, f_emb)

        # Concatenate the two features (features + their embeddings)
        features = torch.hstack((features, features_2))

        x = self.ffn(features)
        out_logits = self.outDense(
            x)
        return out_logits

