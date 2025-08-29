from data_loader import getCytoidLevelChart
from chart_feature_extractor import get_all_feature
import xgboost as xgb
import pickle

model = xgb.Booster()
model.load_model("CytusIIDifficultyAnalyzer_alpha.json")

CHART_FILE_NAME = "dive.astaroth2"
IS_CYTOID = True
# IS_CYTOID = False
is_use_candidate_feature = True
# is_use_candidate_feature = False

if IS_CYTOID:
    GLITCH = getCytoidLevelChart(CHART_FILE_NAME)
else:
    with open(CHART_FILE_NAME) as f:
        GLITCH = json.load(f)

X_target, _ = get_all_feature([GLITCH])
if is_use_candidate_feature:
    with open("candidate_feature_idx.pkl", "rb") as f:
        idxs = pickle.load(f)
    X_subset = X_target[:, idxs]
    X_target = X_subset

y_target = model.predict(xgb.DMatrix(X_target))
print()
print(y_target)
