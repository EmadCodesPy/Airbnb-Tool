import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
JUN_PATH = os.getenv('NOTEBOOK_JUN_PATH')
SEP_PATH = os.getenv('NOTEBOOK_SEP_PATH')
DATA_PATH = os.getenv('DATA_PATH')

def clean_listings(df):
    df = df[df['source'] == 'city scrape'].dropna(subset='price').copy()
    df.loc[:,'price'] = df.loc[:,'price'].str.replace(',','').str.strip('$')
    df['price'] = df['price'].astype(float)
    df['bathroom_shared'] = df['bathrooms_text'].str.contains('shared', case=False, na=False)
    df['bathroom_cleaned'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
    df.loc[df['bathrooms_text'].str.contains('half', case=False, na=False), 'bathroom_cleaned'] = 0.5
    df['bathrooms_final'] = df['bathrooms'].combine_first(df['bathroom_cleaned'])
    df['first_review'] = pd.to_datetime(df['first_review'])
    df['last_review'] = pd.to_datetime(df['last_review'])
    return df

def clean_calender(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

def listings_sanity(df):
    print(f"bathrooms missing {df['bathrooms_final'].isnull().sum()}")
    print(df.loc[df['bathroom_cleaned'].isnull() & df['bathrooms_text'].notnull(), 'bathrooms_text'].unique())
    print(df.loc[df['bathrooms_final'].isnull(), ['bathrooms','bathrooms_text']])

def set_date(df, month):
    if month == 'jun':
        df['snapshot_date'] = '2026-06-15'
        return df
    elif month == 'sep':
        df['snapshot_date'] = '2025-09-11'
        return df

print('Loading June Data...')
jun_listings = pd.read_csv(f'{JUN_PATH}/listings.csv')
jun_calender = pd.read_csv(f'{JUN_PATH}/calendar.csv')
jun_reviews = pd.read_csv(f'{JUN_PATH}/reviews.csv')
jun_listings = jun_listings.set_index('id')

print('Loading September Data...')
sep_listings = pd.read_csv(f'{SEP_PATH}/listings.csv')
sep_calender = pd.read_csv(f'{SEP_PATH}/calendar.csv')
sep_reviews = pd.read_csv(f'{SEP_PATH}/reviews.csv')
sep_listings = sep_listings.set_index('id')

print('Cleaning listings...')
jun_listings = clean_listings(jun_listings)
sep_listings = clean_listings(sep_listings)

print('Sanity Checks...')
listings_sanity(jun_listings)
listings_sanity(sep_listings)

print('Setting listing dates...')
jun_listings = set_date(jun_listings, 'jun')
sep_listings = set_date(sep_listings, 'sep')

print('Merge and concatenate listings...')
merged_listings = pd.merge(jun_listings, sep_listings, on='id', suffixes=('_2026', '_2025'))
concat_listings = pd.concat([jun_listings, sep_listings], axis=0)

print('Clean merged listings')
merged_listings = merged_listings[merged_listings['neighbourhood_cleansed_2026'] == merged_listings['neighbourhood_cleansed_2025']]

print('Clean calenders...')
jun_calender = clean_calender(jun_calender)
sep_calender = clean_calender(sep_calender)

print('Calculate price change...')
merged_listings['price_change'] = merged_listings['price_2026'] - merged_listings['price_2025']

print('Concatenating calenders...')
concat_calenders = pd.concat([jun_calender, sep_calender], axis=0)

print('Storing in parquet format...')
merged_listings.to_parquet(f'{DATA_PATH}/merged_listings.parquet')
concat_listings.to_parquet(f'{DATA_PATH}/concat_listings.parquet')
concat_calenders.to_parquet(f'{DATA_PATH}/concat_calenders.parquet')