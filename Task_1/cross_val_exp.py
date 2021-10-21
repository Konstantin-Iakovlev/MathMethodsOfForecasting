from data_prep import get_data
from hopls import HOPLS, qsquared
import tensorly as tl
import numpy as np
from sklearn.model_selection import KFold
# import optuna
import json

# def objective(trial):
    # R = trial.suggest_int('R', 2, 50)
    # return np.mean(cross_validation(X, Y, R)[1])


def cross_validation(X, Y, R):
    kf = KFold(n_splits=5, shuffle=False)
    hopls_model = HOPLS(R, [10, 3], [10, 3], metric=qsquared)
    train_scores = []
    valid_scores = []
    for train_idx, valid_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[valid_idx]
        Y_train, Y_val = Y[train_idx], Y[valid_idx]
        hopls_model.fit(X_train, Y_train)

        Y_pred_val = hopls_model.predict(X_val, Y_val)[0]
        valid_scores.append(qsquared(Y_val, Y_pred_val))
        
        Y_pred_train = hopls_model.predict(X_train, Y_train)[0]
        train_scores.append(qsquared(Y_train, Y_pred_train))
    return train_scores, valid_scores


def main():
    # getting whole data
    X, Y = get_data()
    print('Shapes are', X.shape, Y.shape)

    # get optimal R (it is 8)
    np.random.seed(1)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=10)
    r_to_scores = dict()
    for R in range(1, 20, 3):
        train_scores, valid_scores = cross_validation(X, Y, R)
        print(f'R = {R},\ntrain: {train_scores},\tvalid: {valid_scores}')
        r_to_scores[R] = [train_scores, valid_scores]


    # save data
    with open('cross_val.json', 'w') as out:
        json.dump(r_to_scores, out)


if __name__ == "__main__":
    main()


