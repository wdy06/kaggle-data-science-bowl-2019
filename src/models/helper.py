from models.model_lgbm import ModelLGBMClassifier, ModelLGBMRegressor
from models.model_catboost import ModelCatBoostRegressor
from models.model_xgb import ModelXGBRegressor

MODEL_MAP = {
    'ModelLGBMClassifier': ModelLGBMClassifier,
    'ModelLGBMRegressor': ModelLGBMRegressor,
    'ModelCatBoostRegressor': ModelCatBoostRegressor,
    'ModelXGBRegressor': ModelXGBRegressor
}
