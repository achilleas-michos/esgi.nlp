from keras import backend as K


def true_pos(y_true, y_pred):
    return K.sum(y_true * K.round(y_pred))


def false_pos(y_true, y_pred):
    return K.sum(y_true * (1. - K.round(y_pred)))


def false_neg(y_true, y_pred):
    return K.sum((1. - y_true) * K.round(y_pred))


def precision(y_true, y_pred):
    return true_pos(y_true, y_pred) / (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))


def recall(y_true, y_pred):
    return true_pos(y_true, y_pred) /  (true_pos(y_true, y_pred) + false_neg(y_true, y_pred))


def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))