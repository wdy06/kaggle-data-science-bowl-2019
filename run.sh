python src/train.py -c configs/002_lgbm_regression.yml -f configs/516_features_list.yml --adjust
python src/train.py -c configs/102_catboost_regression.yml -f configs/516_features_list.yml --adjust
python src/train.py -c configs/300_nn_regression.yml -f configs/516_features_list.yml --high-corr features/to_remove_features.json
