from data_loader import getCytoidLevelChart
from chart_feature_extractor import get_all_feature
import xgboost as xgb

model = xgb.Booster()
model.load_model("CytusIIDifficultyAnalyzer.json")

CHART_FILE_NAME = "delta.palette"
IS_CYTOID = True
# IS_CYTOID = False

if IS_CYTOID:
    GLITCH = getCytoidLevelChart(CHART_FILE_NAME)
else:
    with open(CHART_FILE_NAME) as f:
        GLITCH = json.load(f)

X_target, _ = get_all_feature([GLITCH])
y_target = model.predict(xgb.DMatrix(X_target))
print()
print(y_target)
