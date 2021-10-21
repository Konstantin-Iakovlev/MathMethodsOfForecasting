from data_prep import get_data
from hopls import HOPLS, qsquared
import tensorly as tl
import numpy as np
import json
from cross_val_exp import cross_validation


# getting whole data
X, Y = get_data()
print('Shapes are', X.shape, Y.shape)

np.random.seed(1)
R = 1
sigmas = [0.01 * i for i in range(1, 35, 2)]
sigma_to_scores = dict()
for sigma in sigmas:
    noise_X = np.random.randn(*X.shape) * sigma
    train_scores, valid_scores = cross_validation(X + noise_X,
            Y, R)
    sigma_to_scores[sigma] = [train_scores, valid_scores]

# save data
with open('robustness.json', 'w') as out:
    json.dump(sigma_to_scores, out)
