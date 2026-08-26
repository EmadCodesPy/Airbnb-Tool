import polars as pl

def hedonic_features(df):
    
    property_mapping = {'rental unit': 'home/apt',
        'condo': 'home/apt',
        'home': 'home/apt',
        'loft': 'home/apt',
        'townhouse': 'home/apt',
        'serviced apartment': 'home/apt',
        'guest suite': 'home/apt',
        'guesthouse': 'home/apt',
        'villa': 'home/apt',
        'tiny home': 'home/apt',
        'place': 'home/apt',
        'casa particular': 'home/apt',
        'private room': 'home/apt',
        'vacation home': 'home/apt',
        'bed and breakfast': 'hotel',
        'hotel': 'hotel',
        'boutique hotel': 'hotel',
        'hostel': 'hotel',
        'aparthotel': 'hotel',
        'heritage hotel': 'hotel'
    }
    
    amenities_rename = {
        'central air conditioning': 'air conditioning',
        'private backyard \\u2013 fully fenced': 'backyard',
        'private patio or balcony': 'balcony',
        'patio or balcony': 'balcony'
    }
    amenities_keep = {
        'dedicated workspace',
        'tv',
        'heating',
        'private entrance',
        'outdoor dining area',
        'self check-in',
        'canal view',
        'cleaning available during stay',
        'free parking on premises',
        'elevator',
        'gym',
        'city skyline view',
        'ev charger',
        'backyard',
        'air conditioning',
        'balcony'
    }
    
    df = df.with_columns(
        # Missing indicators
        pl.col('host_is_superhost')
        .is_null()
        .cast(pl.UInt8)
        .alias('host_is_superhost_missing'),
        
        pl.col('instant_bookable')
        .is_null()
        .cast(pl.UInt8)
        .alias('instant_bookable_missing'),
        
        pl.col('license')
        .is_not_null()
        .cast(pl.UInt8)
        .alias('has_license'),
        
        # Binary encoding
        pl.col('bathroom_shared')
        .cast(pl.UInt8)
        .alias('bathroom_shared'),
        
        pl.col('host_is_superhost')
        .replace_strict({"t": 1, "f": 0}, default=0)
        .alias('host_is_superhost'),
        
        pl.col('instant_bookable')
        .replace_strict({"t": 1, "f": 0}, default=0)
        .alias('instant_bookable'),
        
        pl.col('amenities')
        .str.strip_chars('[]')
        .str.replace_all('"', '')
        .str.split(',')
        .list.len()
        .alias('amenities_count'),
        
        pl.col('beds')
        .fill_null(pl.col('accommodates')),
        
        pl.col('bedrooms')
        .fill_null(pl.col('accommodates')),
        
        (pl.col('snapshot_date')
        .cast(pl.Date) - pl.col('first_review')
        .cast(pl.Date))
        .dt.total_days()
        .alias('days_since_first_review'),
        
        (pl.col('snapshot_date')
        .cast(pl.Date) - pl.col('last_review')
        .cast(pl.Date))
        .dt.total_days()
        .alias('days_since_last_review'),
        
        pl.col('first_review')
        .is_null()
        .cast(pl.UInt8)
        .alias('has_no_review'),
        
        pl.col('review_scores_rating')
        .fill_null(pl.col('review_scores_rating').mean()),
        
        pl.col('review_scores_cleanliness')
        .fill_null(pl.col('review_scores_cleanliness').mean()),
        
        pl.col('review_scores_location')
        .fill_null(pl.col('review_scores_location').mean()),
        
        pl.col('reviews_per_month')
        .fill_null(0),
        
        pl.col('price')
        .log()
        .alias('log_price')
    )
    
    df = df.with_columns(
        pl.col('days_since_first_review')
        .fill_null(99999),
        
        pl.col('days_since_last_review')
        .fill_null(99999)
    )
    
    # Remove redundant wording in property_type
    regex = r"^(Entire |Private room in |Shared room in |Room in )"
    df = df.with_columns(
        pl.col('property_type')
        .str.replace(regex, '')
        .str.to_lowercase()
    )
    
    # Map the property_type to a category
    df = df.with_columns(pl.col('property_type').replace_strict(property_mapping, default='unique'))

    # Create dummies for neccessary columns (One hot encoding)
    property_dummies = (
        df.select('property_type')
        .to_dummies()
    )
    property_dummies = property_dummies.drop('property_type_home/apt')
    
    neighbourhood_dummies = (
        df.select('neighbourhood_cleansed')
        .to_dummies()
        .rename(lambda x: x.replace('neighbourhood_cleansed_', 'neighbourhood_'))
    )
    neighbourhood_dummies = neighbourhood_dummies.drop('neighbourhood_Centrum-West')
    
    room_type_dummies = (
        df.select('room_type')
        .to_dummies()
    )
    room_type_dummies = room_type_dummies.drop('room_type_Entire home/apt')
    
    # Clean and and turn amenities into list
    df = df.with_columns(
        pl.col('amenities')
        .str.strip_chars('[]')
        .str.replace_all('"', '')
        .str.split(',')
        .list.eval(
            pl.element()
            .str.strip_chars_start()
            )
        .alias('amenities')
    )
    
    # Keep only neccesary entries from amenities
    df = df.with_columns(
        pl.col('amenities')
        .list.eval(
            pl.element()
            .str.to_lowercase()
            .replace(amenities_rename)
            )
        .list.set_intersection(amenities_keep)
        .alias('amenities_clean')
    )
    
    # Store expressions to optimize / One hot encoding amenities
    expressions = []
    for i in amenities_keep:
        expressions.append(
            pl.col('amenities_clean')
            .list.contains(i)
            .cast(pl.UInt8)
            .alias(f'has_{i}'.replace(' ', '_'))
            )
        
    df = df.with_columns(expressions)
    
    # Add the dummies at the end to the dataframe
    df = pl.concat(
        [df, room_type_dummies, neighbourhood_dummies, property_dummies],
        how='horizontal_extend'
    )
    
    # Drop the un-encoded columns
    
    # Filter
    df = df.filter(
        (pl.col('minimum_nights').is_not_null()) & (pl.col('bathrooms_final').is_not_null()) #& (pl.col('price') < 1498.0) # & (pl.col('latitude').is_not_null()) & (pl.col('longitude').is_not_null())
    )
    # Rename columns
    df = df.rename(lambda col: (col.lower().replace(" ", "_")))
    
    return df