# KFold 測試
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from chart_feature_extractor import getXYdata


def doKFoldTest(X, Y, model, n_splits):

    # 設定折數
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test = y_test.reshape(-1)
        y_pred = y_pred.reshape(-1)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    return round(np.mean(r2_scores), 4)


if __name__ == "__main__":
    X, Y, feature_name, all_song_name = getXYdata(True)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    doKFoldTest(X, Y, model, n_splits=10)
