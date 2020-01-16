from models.model_lgbm import ModelLGBMClassifier, ModelLGBMRegressor
from models.model_catboost import ModelCatBoostRegressor

MODEL_MAP = {
    'ModelLGBMClassifier': ModelLGBMClassifier,
    'ModelLGBMRegressor': ModelLGBMRegressor,
    'ModelCatBoostRegressor': ModelCatBoostRegressor
}
