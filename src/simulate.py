import polars as pl
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')  # wherever the joblib folder lives

# --- Load everything once ---
occupancy_model = joblib.load(f'{MODEL_PATH}/occupancy_model_xgb.joblib')
local_groups = joblib.load(f'{DATA_PATH}/local_groups.joblib')
citywide_groups = joblib.load(f'{DATA_PATH}/citywide_groups.joblib')
full_df = pl.read_parquet(f'{DATA_PATH}/occupancy_features_full.parquet')

# Must match the exact FEATURES list/order used to train occupancy_model
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

OCCUPANCY_FEATURES = [
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
    OCCUPANCY_FEATURES.append(f'has_{i}')
for i in neighbourhoods_keep:
    OCCUPANCY_FEATURES.append(f'neighbourhood_{i}')



SNAPSHOT = '2026-06-15'  # pin simulation to most recent snapshot — confirm this matches your actual date format

PLACEHOLDER_REVIEW_TO_BOOKING = 2.0  # TODO: replace with real calibration later


def percentile_rank(x, others, comp_size):
    """Replicates polars rank(method='average') behavior for inserting x into others."""
    if len(others) == 0 or comp_size == 0:
        return 0.0
    count_less = np.sum(others < x)
    count_equal = np.sum(others == x) + 1  # +1 for x itself
    rank = count_less + (count_equal + 1) / 2
    return rank / comp_size


def get_listing_row(listing_id, snapshot_date=SNAPSHOT):
    row = full_df.filter(
        (pl.col('id').cast(pl.Utf8) == str(listing_id)) & (pl.col('snapshot_date') == snapshot_date)
    )
    if row.height == 0:
        raise ValueError(f'No row found for listing {listing_id} at snapshot {snapshot_date}')
    if row.height > 1:
        raise ValueError(f'Multiple rows found for listing {listing_id} at snapshot {snapshot_date} — expected exactly one')
    return row.to_dicts()[0]


def get_price_bounds(listing_row, lower_q=0.02, upper_q=0.95):
    """
    Bound the sweep to the empirical price range of the listing's comp group,
    so the optimizer can't recommend prices the model has no real basis to evaluate.
    Falls back to citywide (room_type + bucket) comps if the local group is too small.
    """
    if listing_row['local_comp'] == 1:
        comps = full_df.filter(
            (pl.col('neighbourhood_cleansed') == listing_row['neighbourhood_cleansed']) &
            (pl.col('room_type') == listing_row['room_type']) &
            (pl.col('bucket') == listing_row['bucket'])
        ).select('price').to_series()
    else:
        comps = full_df.filter(
            (pl.col('room_type') == listing_row['room_type']) &
            (pl.col('bucket') == listing_row['bucket'])
        ).select('price').to_series()

    return comps.quantile(lower_q), comps.quantile(upper_q)


def simulate_revenue_curve(listing_id, snapshot_date=SNAPSHOT, grid_step=10,
                            lower_q=0.02, upper_q=0.95, smooth_window=5):
    listing_row = get_listing_row(listing_id, snapshot_date)

    # --- Bound the sweep BEFORE simulating, not after ---
    lower_bound, upper_bound = get_price_bounds(listing_row, lower_q, upper_q)
    price_grid = np.arange(lower_bound, upper_bound, grid_step)

    predicted_value = listing_row['predicted_value']
    key_local = (listing_row['neighbourhood_cleansed'], listing_row['room_type'], listing_row['bucket'])
    key_city = (listing_row['room_type'], listing_row['bucket'])

    local_arr = local_groups.get(key_local)
    city_arr = citywide_groups.get(key_city)

    if local_arr is not None:
        local_others = local_arr[local_arr[:, 0] != listing_id, 1].astype(float)
    else:
        local_others = np.array([])

    if city_arr is not None:
        city_others = city_arr[city_arr[:, 0] != listing_id, 1].astype(float)
    else:
        city_others = np.array([])

    rows = []
    for p in price_grid:
        value_gap_p = predicted_value - p

        local_pct = (percentile_rank(value_gap_p, local_others, listing_row['local_comp_size'])
                     if listing_row['local_comp'] == 1 else 0.0)
        city_pct = (percentile_rank(value_gap_p, city_others, listing_row['citywide_comp_size'])
                    if listing_row['citywide_comp'] == 1 else 0.0)

        feat = dict(listing_row)
        feat['value_gap'] = value_gap_p
        feat['local_percentile_rank'] = local_pct
        feat['citywide_percentile_rank'] = city_pct
        feat['_price'] = p
        rows.append(feat)

    sim_df = pd.DataFrame(rows)
    X_sim = sim_df[OCCUPANCY_FEATURES]
    predicted_reviews = occupancy_model.predict(X_sim)

    expected_bookings_30d = predicted_reviews * PLACEHOLDER_REVIEW_TO_BOOKING
    expected_revenue_30d = sim_df['_price'].to_numpy() * expected_bookings_30d

    curve = pd.DataFrame({
        'price': sim_df['_price'],
        'predicted_reviews_per_month': predicted_reviews,
        'expected_bookings_30d': expected_bookings_30d,
        'expected_revenue_30d': expected_revenue_30d,
    })

    # --- Smooth before picking the max — raw argmax is noisy due to tree splits ---
    curve['expected_revenue_30d_smoothed'] = (
        curve['expected_revenue_30d']
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )

    return curve


def recommend_price(listing_id, snapshot_date=SNAPSHOT, **kwargs):
    curve = simulate_revenue_curve(listing_id, snapshot_date, **kwargs)
    best_idx = curve['expected_revenue_30d_smoothed'].idxmax()
    best_price = curve.loc[best_idx, 'price']

    price_range = curve['price'].max() - curve['price'].min()
    dist_from_top = curve['price'].max() - best_price
    bound_constrained = dist_from_top < 0.1 * price_range

    return {
        'recommended_price': best_price,
        'expected_revenue_30d': curve.loc[best_idx, 'expected_revenue_30d_smoothed'],
        'bound_constrained': bound_constrained,
        'curve': curve,
    }

def run_backtest(n_sample=50, snapshot_date=SNAPSHOT, random_state=369):
    listing_ids = (
        full_df.filter(pl.col('snapshot_date') == snapshot_date)
        .select('id')
        .to_series()
        .sample(n=n_sample, seed=random_state)
    )

    results = []
    for lid in listing_ids:
        try:
            listing_row = get_listing_row(lid, snapshot_date)
            actual_price = listing_row['price']
            actual_reviews = listing_row['reviews_per_month']  # ground truth demand
            actual_revenue = actual_price * actual_reviews * PLACEHOLDER_REVIEW_TO_BOOKING

            rec = recommend_price(lid, snapshot_date)

            lift_pct = (
                (rec['expected_revenue_30d'] - actual_revenue) / actual_revenue * 100
                if actual_revenue > 0 else np.nan
            )

            results.append({
                'listing_id': lid,
                'actual_price': actual_price,
                'actual_revenue_30d': actual_revenue,
                'recommended_price': rec['recommended_price'],
                'expected_revenue_30d': rec['expected_revenue_30d'],
                'lift': rec['expected_revenue_30d'] - actual_revenue,
                'lift_pct': lift_pct,
                'bound_constrained': rec['bound_constrained'],
            })
        except Exception as e:
            print(f'Skipped listing {lid}: {e}')
            continue

    return pd.DataFrame(results)


if __name__ == '__main__':
    bt = run_backtest(n_sample=200)
    print(bt.describe())
    print(f"\nMedian lift %: {bt['lift_pct'].median():.1f}%")
    print(f"Mean lift %: {bt['lift_pct'].mean():.1f}%")
    print(f"% bound-constrained: {(bt['bound_constrained'].sum() / len(bt)) * 100:.1f}%")
    

# if __name__ == '__main__':
#     sample_id = full_df.filter(pl.col('snapshot_date') == SNAPSHOT).select('id').to_series()[2]
#     actual_price = full_df.filter(
#         (pl.col('id') == sample_id) & (pl.col('snapshot_date') == SNAPSHOT)
#     ).select('price').item()

#     result = recommend_price(sample_id)

#     print(result['curve'].to_string())
#     print(f"\nActual price: {actual_price}")
#     print(f"Recommended price (smoothed): {result['recommended_price']:.2f}")
#     print(f"Expected 30d revenue at recommended price: {result['expected_revenue_30d']:.2f}")