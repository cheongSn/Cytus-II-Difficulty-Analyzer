# 訓練
# from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from chart_feature_extractor import getXYdata
from utils import pad_display
import numpy as np

X, Y, feature_name, all_song_name = getXYdata(is_use_cache=True)

# not use
candidate = [
    "burst_p90",
    "page_space_p90_score",
    "page_space_third_score",
    "burst_song_avg",
    "complex_beat_count",
    "Drag-child",
    "burst_LR_low_max",
    "double_count",
    "SONG_LENGTH",
    "burst_endurance_8",
    "CDrag-head",
    "MAIN_BPM",
    "burst_fifth",
]
feat2idx = {f: i for i, f in enumerate(feature_name)}
idxs = [feat2idx[feat] for feat in candidate]
X_subset = X[:, idxs]

# 1. 分割訓練與測試集
used_data_size = None
indices = np.arange(len(X[:used_data_size]))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, Y, indices, test_size=0.1, random_state=75
)
# X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
#     X_subset, Y, indices, test_size=0.1, random_state=75
# )

print(f"Train size:{len(X_train)}")
print(f"Test size:{len(X_test)}")

# model = RandomForestRegressor(n_estimators=100, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test = y_test.reshape(-1)
y_pred = y_pred.reshape(-1)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

data = []
for i, idx in enumerate(test_idx):
    song_name = all_song_name[idx]
    data.append(
        (
            idx,
            song_name,
            y_test[i],
            round(y_pred[i], 2),
            round(y_pred[i] - y_test[i], 2),
        )
    )

SORT_BY = 4
# SORT_BY = 3

data = sorted(data, key=lambda x: (x[SORT_BY]))
print()
print(f"ID\t{pad_display('曲名', 32)}\t難度\t等級\t預測\t誤差")
for idx, s, t, p, diff in data:
    print(f"{idx}\t{pad_display(s[:30], 35)}\t{t}\t{p}\t{diff}")

# y_pred = np.round(y_pred, 2)
# print("\t".join(map(str, np.reshape(y_pred, (-1,)))))
# print("\t".join(map(str, np.reshape(y_test, (-1,)))))

print()
print("TEST Mean Squared Error:", mse)
print("TEST R² Score:", r2)
print("===================================================================")

if False:
    y_pred = model.predict(X_train)
    y_train = y_train.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    data = []
    for i, idx in enumerate(train_idx):
        song_name = all_song_name[idx]
        data.append(
            (
                idx,
                song_name,
                y_train[i],
                round(y_pred[i], 2),
                round(y_pred[i] - y_train[i], 2),
            )
        )

    data = sorted(data, key=lambda x: (x[5]))
    print()
    print(f"ID\t{pad_display('曲名', 32)}\t難度\t等級\t預測\t誤差")
    for idx, s, t, p, diff in data:
        print(f"{idx}\t{pad_display(s[:30], 35)}\t{t}\t{p}\t{diff}")

    # y_pred = np.round(y_pred, 2)
    # print("\t".join(map(str, np.reshape(y_pred, (-1,)))))
    # print("\t".join(map(str, np.reshape(y_test, (-1,)))))

    print()
    print("TRAIN Mean Squared Error:", mse)
    print("TRAIN R² Score:", r2)


model.save_model("CytusIIDifficultyAnalyzer_alpha.json")
