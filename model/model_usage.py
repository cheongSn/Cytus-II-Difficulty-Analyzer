from data_loader import getCytoidLevelChart
from chart_feature_extractor import get_all_feature
import xgboost as xgb
import pickle


IS_CYTOID = True
# IS_CYTOID = False
IS_ALPHA = True
IS_ALPHA = False
CHART_FILE_NAME = "dive.astaroth2"
CHART_FILE_NAME = "delta.palette"

model = xgb.Booster()
if IS_ALPHA:
    model.load_model("CytusIIDifficultyAnalyzer_alpha.json")
    with open("candidate_feature_idx_alpha.pkl", "rb") as f:
        idxs = pickle.load(f)
else:
    model.load_model("CytusIIDifficultyAnalyzer.json")
    with open("candidate_feature_idx.pkl", "rb") as f:
        idxs = pickle.load(f)

if IS_CYTOID:
    GLITCH = getCytoidLevelChart(CHART_FILE_NAME)
else:
    with open(CHART_FILE_NAME) as f:
        GLITCH = json.load(f)

X_target, _ = get_all_feature([GLITCH])
X_target = X_target[:, idxs]

y_target = model.predict(xgb.DMatrix(X_target))
print(y_target)
