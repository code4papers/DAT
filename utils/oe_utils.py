import tensorflow as tf
import keras.backend as K
import numpy as np
from scipy.special import logsumexp
from scipy.special import softmax
from sklearn.metrics import roc_curve,auc,roc_auc_score,average_precision_score


def my_ood_loss(y_true, y_predict):
    num_in = int(len(y_true) / 2)
    loss = tf.keras.losses.categorical_crossentropy(y_true[:num_in], y_predict[:num_in])
    loss += 0.5 * -(tf.reduce_mean(y_predict[num_in:], 1) - tf.reduce_mean(tf.math.reduce_logsumexp(y_predict[num_in:], axis=1)))
    return loss


def calculate_oe_score(logit_layer, x_data, use_xent=False):
    _scores = []
    _preds = []
    _top2_diff = []

    # logit_layer = K.function(inputs=model.input, outputs=model.layers[-2].output)
    output = logit_layer([x_data])
    print(output.shape)
    smax = softmax(output, axis=1)
    _preds = np.argmax(output, axis=1)

    temp = np.sort(-1 * smax)
    top2_diff = -1 * (temp[:, 0] - temp[:, 1])
    _top2_diff.append(top2_diff)
    if use_xent:
        _scores = np.mean(output, axis=1) - logsumexp(output, axis=1)
    else:
        _scores = 1 - np.max(output, axis=1)

    return _top2_diff, _scores, _preds


def merge_and_generate_labels(X_pos, X_neg):

    X = np.concatenate((X_pos, X_neg))
    if len(X.shape) == 1:
        X = X.reshape((X.shape[0], -1))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    return X, y


def get_metric_scores(y_true, y_score, tpr_level):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    tpr95_pos = np.abs(tpr - tpr_level).argmin()
    tnr_at_tpr95 = 1. - fpr[tpr95_pos]
    aupr = average_precision_score(y_true, y_score)
    results = {"TNR": tnr_at_tpr95, 'AUROC': auroc, 'AUPR': aupr, "TNR_threshold": thresholds[tpr95_pos],
               'FPR': fpr, 'TPR': tpr, "threshold": thresholds}

    return results


def block_split(X, Y, train_num, partition = 5000):

    X_pos, Y_pos = X[:partition], Y[:partition]
    X_neg, Y_neg = X[partition:], Y[partition:]
    np.random.seed(0)
    random_index = np.arange(partition)
    np.random.shuffle(random_index)
    X_pos, Y_pos = X_pos[random_index], Y_pos[random_index]
    X_neg, Y_neg = X_neg[random_index], Y_neg[random_index]

    X_train = np.concatenate((X_pos[:train_num], X_neg[:train_num]))
    Y_train = np.concatenate((Y_pos[:train_num], Y_neg[:train_num]))
    X_test = np.concatenate((X_pos[train_num:], X_neg[train_num:]))
    Y_test = np.concatenate((Y_pos[train_num:], Y_neg[train_num:]))
    return X_train, Y_train, X_test, Y_test


