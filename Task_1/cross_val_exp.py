from data_prep import get_data
from hopls import HOPLS, qsquared
import tensorly as tl
import numpy as np
from sklearn.model_selection import KFold
import optuna
import json



def cross_validation(X, Y, R, ranks=[1, 1]):
    kf = KFold(n_splits=5, shuffle=False)
    hopls_model = HOPLS(R, ranks, ranks, metric=qsquared)
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
    # def objective(trial):
        # r1 = trial.suggest_int('r1', 1, 30)
        # r2 = trial.suggest_int('r2', 1, 8)
        # return np.mean(cross_validation(X, Y, 4, ranks=(r1, r2))[1])
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=40)
    r_to_scores = dict()
    for R in [1, 4, 10, 13, 16, 19]:
        train_scores, valid_scores = cross_validation(X, Y, R, ranks=(3, 1))
        # print(f'R = {R},\ntrain: {train_scores},\tvalid: {valid_scores}')
        print(f'R = {R}, \ntrain: {np.mean(train_scores), np.std(train_scores)}, \t\
                valid: {np.mean(valid_scores), np.std(valid_scores)}')
        r_to_scores[R] = [train_scores, valid_scores]


    # # save data
    with open('cross_val.json', 'w') as out:
        json.dump(r_to_scores, out)


if __name__ == "__main__":
    main()


