import polars as pl

def occupancy_features(df, predictions_df):
    
    buckets = {
    1: '1-2',
    2: '1-2',
    3: '3-4',
    4: '3-4',
    5: '5-6',
    6: '5-6',
    7: '7-8',
    8: '7-8'
    }
    # Assign buckets to rows
    df = df.with_columns(
        pl.col('accommodates')
        .replace_strict(buckets, default='9+')
        .alias('bucket'),
        
        (1 - pl.col('availability_30')
        .truediv(30))
        .alias('occupancy_rate')
    )
    # Identify and save whether each row can be compared to ints neighbours
    comp_group = df.group_by(['neighbourhood_cleansed', 'room_type', 'bucket']).agg(pl.len())
    comp_group = comp_group.with_columns(
        pl.when(pl.col('len')>20)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('local_comp')
    )
    
    # Add comp info back to the dataframe
    df = df.join(comp_group, on=['neighbourhood_cleansed', 'room_type', 'bucket'], how='left').rename({'len': 'local_comp_size'})

    comp_group = df.group_by(['room_type', 'bucket']).agg(pl.len())
    comp_group = comp_group.with_columns(
        pl.when(pl.col('len')>20)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('citywide_comp')
    )

    df = df.join(comp_group, on=['room_type', 'bucket'], how='left').rename({'len': 'citywide_comp_size'})


    # Add predictions to the dataframe
    df = df.join(predictions_df, on=['id', 'snapshot_date'], how='left')
    
    df = df.with_columns(
        (pl.col('predicted_value') - pl.col('price'))
        .alias('value_gap')
    )
    
    df = df.with_columns(
        
        (pl.col('value_gap')
        .rank(method='average', descending=False)
        .over(['room_type', 'bucket']) / pl.col('citywide_comp_size'))
        .alias('citywide_percentile_rank'),
        
        (pl.col('value_gap')
        .rank(method='average', descending=False)
        .over(['neighbourhood_cleansed', 'room_type', 'bucket']) / pl.col('local_comp_size'))
        .alias('local_percentile_rank')
    )
    
    df = df.with_columns(
        
        pl.when(pl.col('citywide_comp') == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col('citywide_percentile_rank'))
            .alias('citywide_percentile_rank'),
        
        pl.when(pl.col('local_comp') == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col('local_percentile_rank'))
            .alias('local_percentile_rank')
    )
    
    return df


def build_comp_cache(df):
    local_groups = {}
    for key, group in df.group_by(['neighbourhood_cleansed', 'room_type', 'bucket']):
        local_groups[key] = group.select(['id', 'value_gap']).to_numpy()

    citywide_groups = {}
    for key, group in df.group_by(['room_type', 'bucket']):
        citywide_groups[key] = group.select(['id', 'value_gap']).to_numpy()

    return local_groups, citywide_groups

if __name__ == '__main__':
    import joblib
    from dotenv import load_dotenv
    import os
    from features_hedonic import hedonic_features
    
    load_dotenv()
    LISTINGS_DATA = os.getenv('CONCAT_LISTINGS_PATH')
    PRED_DATA = os.getenv('PRED_PATH')
    DATA_PATH = os.getenv('DATA_PATH')

    ldf = pl.read_parquet(LISTINGS_DATA)
    predictions = pl.read_parquet(PRED_DATA)
    
    ldf = hedonic_features(ldf)
    ldf = occupancy_features(ldf, predictions)
    
    local_groups, citywide_groups = build_comp_cache(ldf)
    joblib.dump(local_groups, f'{DATA_PATH}/local_groups.joblib')
    joblib.dump(citywide_groups, f'{DATA_PATH}/citywide_groups.joblib')
    
    ldf.write_parquet(f'{DATA_PATH}/occupancy_features_full.parquet')