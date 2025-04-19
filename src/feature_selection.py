import os
import json
import pandas as pd
import numpy as np
import catboost as cb
from utils import load_config
from sklearn.feature selection import SelectFromModel, mutual_information_regression

if __name__ == '__main__':
    # get config and data
    config = load_config()

    DATA_DIR = config['final_data']
    TRAIN = os.path.join(DATA_DIR, 'train.csv')

    CATEGORICAL = ['river', 'location', 'month', 'week', 'season', 'station_code']
    NUM_SOIL = ['bdod', 'cfvo', 'clay', 'sand']
    NUM_METEO = ['tp', 't2m', 'swvl1', 'evap']
    COLS_TO_DROP = [
        'ObsDate', 'catchment', 'hydro_region', 'hydro_sector', 'hydro_sub_sector', 'hydro_zone', 'region_sector', 
        'region_sub_sector', 'region_zone', 'sector_sub_sector', 'sector_zone', 'sub_sector_zone'
    ]
    TARGET_COLS = ['water_flow_week_1', 'water_flow_week_2', 'water_flow_week_3', 'water_flow_week_4']


    df = pd.read_csv(TRAIN)
    X_ = df.drop(columns = TARGET_COLS + COLS_TO_DROP + CATEGORICAL, errors = 'ignore')
    y_= df.water_flow_week_1

    selected_features = []
    for feature_group in FEATURE_GROUPS:
        XX = X_.filter(regex = '|'.join(feature_group))
        features = np.array(XX.columns)
        mi = mutual_info_regression(XX, y_)
        select_idx = np.where(mi > np.quantile(mi, 0.5))[0]
        feature_imp = mi[select_idx]
        features = features[select_idx]
        features_ranked = sorted(zip(features, feature_imp), key = lambda x: x[1], reverse = True)
        selected_features.extend(features_ranked)
    selected_features_df = pd.DataFrame(selected_features, columns = ['feature', 'y_corr'])

    new_features = selected_features_df.feature.tolist()
    new_features = new_features + [i for i in X_.columns if 'water_flow' in i]
    X_new = X_[new_features]

    np.random.seed(0)
    reg = cb.CatBoostRegressor(
        iterations = 100, 
        depth = 6,
        objective = 'MAE', 
        eval_metric = 'MAE', 
        verbose = 0, 
        random_state = 0, 
        thread_count=1, 
        task_type="CPU"
    )

    sfm = SelectFromModel(reg, threshold="median", max_features = 20)
    sfm.fit(X_new, y_)

    selected_features = pd.DataFrame({'features': sfm.get_feature_names_out()})
    selected_features.to_csv(os.path.join(DATA_DIR, 'selected_features.csv'), index = False)
    
    
    
    