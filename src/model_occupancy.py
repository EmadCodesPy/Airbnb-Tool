from sklearn.linear_model import Lasso, Ridge, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, make_scorer
from dotenv import load_dotenv
import os
import polars as pl
from features_hedonic import hedonic_features
from features_occupancy import occupancy_features
import numpy as np
import pandas as pd
from xgboost import XGBModel

load_dotenv()
LISTINGS_DATA = os.getenv('CONCAT_LISTINGS_PATH')
CALENDER_DATA = os.getenv('CONCAT_CALENDER_PATH')
DATA_PATH = os.getenv('DATA_PATH')
PRED_DATA = os.getenv('PRED_PATH')


print('Loading parquet data...')
ldf = pl.read_parquet(LISTINGS_DATA)
predictions = pl.read_parquet(PRED_DATA)
print(f'Dataframe shape = {ldf.shape}')
print('Commencing feature engineering...')
ldf = hedonic_features(ldf)
ldf = occupancy_features(ldf, predictions)
print(f'Feature engineered dataframe shape = {ldf.shape}')

amenities_keep = [
    'city_skyline_view',
    'dedicated_workspace',
    'elevator',
    'gym',
    'canal_view',
    'outdoor_dining_area',
    'tv',
    'cleaning_available_during_stay',
    'balcony',
    'backyard',
    'free_parking_on_premises',
    'air_conditioning'
]

neighbourhoods_keep = [
    'ijburg_-_zeeburgereiland',
    'noord-oost',
    'de_baarsjes_-_oud-west',
    'gaasperdam_-_driemond',
    'bos_en_lommer',
    'bijlmer-centrum',
    'slotervaart',
    'oud-oost',
    'bijlmer-oost',
    'watergraafsmeer',
    'noord-west',
    'de_aker_-_nieuw_sloten',
    'zuid',
    'centrum-oost',
    'geuzenveld_-_slotermeer',
    'osdorp',
    'westerpark',
    'de_pijp_-_rivierenbuurt',
    'oostelijk_havengebied_-_indische_buurt',
    'oud-noord',
    'buitenveldert_-_zuidas'
]

FEATURES = [
    'host_is_superhost',
    'accommodates',
    'has_no_review',
    'beds',
    'bedrooms',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_location',
    'instant_bookable',
    'bathrooms_final',
    'instant_bookable_missing',
    'host_is_superhost_missing',
    # 'room_type_hotel_room',
    'room_type_private_room',
    'room_type_shared_room',
    # 'property_type_hotel',
    'local_comp',
    'citywide_comp',
    'value_gap',
    'citywide_percentile_rank',
    'local_percentile_rank'
]
for i in amenities_keep:
    FEATURES.append(f'has_{i}')
for i in neighbourhoods_keep:
    FEATURES.append(f'neighbourhood_{i}')

TARGET = 'reviews_per_month'

ldf = ldf.filter(pl.col('property_type_hotel') == 0)

X = ldf.select(FEATURES).to_pandas()
y = ldf.select(TARGET).to_pandas().squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=369
)


models = []

print('Fitting models...')
model_LR = Ridge()
model_LR.fit(X_train, y_train)

model_XG = XGBModel()
model_XG.fit(X_train, y_train)
models.append(model_LR)
models.append(model_XG)

def rate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rate_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

scoring = {
    "r2": "r2",
    "mae_price": make_scorer(rate_mae),
    "rmse_price": make_scorer(rate_rmse),
}

import joblib
MODEL_PATH = os.getenv('MODEL_PATH')
joblib.dump(model_XG, f'{MODEL_PATH}/occupancy_model_xgb.joblib')


print('Predicting...')
for model in models:

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False,
    )

    y_pred = model.predict(X_test)

    print(f"{model} - R²:         {scores['test_r2'].mean():.3f}")
    print(f"{model} - Reviews MAE:  {scores['test_mae_price'].mean():.2f}")
    print(f"{model} - Reviews RMSE: {scores['test_rmse_price'].mean():.2f}")

# if __name__ == '__main__':
#     predictions = model_XG.predict(X)
#     ldf = ldf.with_columns(pl.Series('predicted_value', (predictions)))
#     df = ldf[['id', 'snapshot_date', 'predicted_value']]
#     df.write_parquet(f'{DATA_PATH}/hedonic_predictions.parquet')
#     print('Done')
    

#     ('host_is_superhost', String),
#     ('neighbourhood_cleansed', String),
#     ('property_type', String),
#     ('room_type', String),
#     ('accommodates', Int64),
    # 'bathrooms_final',
    # has_no_reviews
    # number_of_reviews
#     ('bedrooms', Float64),
#     ('beds', Float64),
#     ('amenities', String),
#     ('instant_bookable', String),
#     ('bathrooms_final', Float64),
#     ('occupancy_rate', Float64),
#     ('local_comp', Int32),
#     ('citywide_comp', Int32),
#     ('value_gap', Float64),
#     ('citywide_percentile_rank', Float64),
#     ('local_percentile_rank', Float64),
        # ('review_scores_rating', Float64),
        # ('review_scores_cleanliness', Float64),
        # ('review_scores_location', Float64),
            # 'instant_bookable_missing',
            # host_is_superhost_missing



# SEASONALITY INCLUDE USING SNAPSHOT DATE