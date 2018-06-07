import numpy as np
from keras import backend as K

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        if x.shape[1] != 3:
            return np.array([-1])
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        if x.shape[1] != 3:
            return np.array([-1])
        x[:, :, :, 0] -= 104.006
        x[:, :, :, 1] -= 116.669
        x[:, :, :, 2] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x
