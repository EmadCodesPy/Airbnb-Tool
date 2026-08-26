from sklearn.linear_model import Lasso, Ridge, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, make_scorer
from dotenv import load_dotenv
import os
import polars as pl
from features_hedonic import hedonic_features
import numpy as np
import pandas as pd
from xgboost import XGBModel
import joblib

load_dotenv()
LISTINGS_DATA = os.getenv('CONCAT_LISTINGS_PATH')
CALENDER_DATA = os.getenv('CONCAT_CALENDER_PATH')
DATA_PATH = os.getenv('DATA_PATH')

print('Loading parquet data...')
ldf = pl.read_parquet(LISTINGS_DATA)
print(f'Dataframe shape = {ldf.shape}')
print('Commencing feature engineering...')
ldf = hedonic_features(ldf)
print(f'Feature engineered dataframe shape = {ldf.shape}')

# amenities_keep = [
#     'city_skyline_view',
#     'heating',
#     'dedicated_workspace',
#     'elevator',
#     'ev_charger',
#     'gym',
#     'canal_view',
#     'outdoor_dining_area',
#     'private_entrance',
#     'tv',
#     'cleaning_available_during_stay',
#     'balcony',
#     'self_check-in',
#     'backyard',
#     'free_parking_on_premises',
#     'air_conditioning'
# ]

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

# FEATURES = [
#     'host_is_superhost',
#     'accommodates',
#     'beds',
#     'bedrooms',
#     'number_of_reviews',
#     'review_scores_rating',
#     'review_scores_cleanliness',
#     'review_scores_location',
#     'instant_bookable',
#     'reviews_per_month',
#     'bathroom_shared',
#     'bathrooms_final',
#     'host_is_superhost_missing',
#     'has_license',
#     'instant_bookable_missing',
#     'amenities_count',
#     'days_since_first_review',
#     'days_since_last_review',
#     'has_no_review',
#     'room_type_hotel_room',
#     'room_type_private_room',
#     'room_type_shared_room',
#     'property_type_hotel',
#     'property_type_unique'
# ]

FEATURES = [
    'host_is_superhost',
    'accommodates',
    'beds',
    'bedrooms',
    'number_of_reviews',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_location',
    'instant_bookable',
    'bathroom_shared',
    'bathrooms_final',
    'instant_bookable_missing',
    'days_since_first_review',
    'days_since_last_review',
    'room_type_hotel_room',
    'room_type_private_room',
    'room_type_shared_room',
    'property_type_hotel',
]
for i in amenities_keep:
    FEATURES.append(f'has_{i}')
for i in neighbourhoods_keep:
    FEATURES.append(f'neighbourhood_{i}')


TARGET = 'log_price'

X = ldf.select(FEATURES).to_pandas()
y = ldf.select(TARGET).to_pandas().squeeze()
price = ldf.select('price').to_pandas().squeeze()  # keep raw price around for the cutoff step

X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
    X, y, price, test_size=0.2, random_state=369
)

cutoff = price_train.quantile(0.995)  # computed on TRAIN ONLY

train_mask = price_train <= cutoff
X_train, y_train = X_train[train_mask], y_train[train_mask]

test_mask = price_test <= cutoff  # same fixed cutoff, applied without recomputing it
X_test, y_test = X_test[test_mask], y_test[test_mask]



models = []

print('Fitting models...')
model_LR = Ridge()
model_LR.fit(X_train, y_train)

params = {
    'max_depth': 8,
    
}
model_XG = XGBModel()
model_XG.fit(X_train, y_train)
models.append(model_LR)
models.append(model_XG)

def price_mae(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

def price_rmse(y_true, y_pred):
    return root_mean_squared_error(np.exp(y_true), np.exp(y_pred))

scoring = {
    "r2": "r2",
    "mae_log": "neg_mean_absolute_error",
    "rmse_log": "neg_root_mean_squared_error",
    "mae_price": make_scorer(price_mae),
    "rmse_price": make_scorer(price_rmse),
}
MODEL_PATH = os.getenv('MODEL_PATH')
joblib.dump(model_XG, f'{MODEL_PATH}/hedonic_model_xgb.joblib')

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
    print(f"{model} - Log MAE:    {-scores['test_mae_log'].mean():.3f}")
    print(f"{model} - Log RMSE:   {-scores['test_rmse_log'].mean():.3f}")
    print(f"{model} - Price MAE:  {scores['test_mae_price'].mean():.2f}")
    print(f"{model} - Price RMSE: {scores['test_rmse_price'].mean():.2f}")

if __name__ == '__main__':
    predictions = model_XG.predict(X)
    ldf = ldf.with_columns(pl.Series('predicted_value', np.exp(predictions)))
    df = ldf[['id', 'snapshot_date', 'predicted_value']]
    df.write_parquet(f'{DATA_PATH}/hedonic_predictions.parquet')
    print('Done')