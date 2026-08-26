# import streamlit as st
# import polars as pl
# import pandas as pd
# import plotly.graph_objects as go
# from simulate import (
#     full_df, get_listing_row, get_price_bounds,
#     simulate_revenue_curve, recommend_price, SNAPSHOT
# )

# st.set_page_config(page_title="Amsterdam Airbnb Pricing Engine", layout="wide")

# st.title("Amsterdam Airbnb Dynamic Pricing")
# st.caption("Hedonic pricing + comp-based occupancy modeling — recommendation, not causal elasticity")

# # --- Listing selector ---
# listing_options = (
#     full_df.filter(pl.col('snapshot_date') == SNAPSHOT)
#     .select(['id', 'neighbourhood_cleansed', 'room_type', 'price'])
#     .to_pandas()
# )
# listing_options['label'] = (
#     listing_options['id'].astype(str) + " — " +
#     listing_options['neighbourhood_cleansed'] + " — " +
#     listing_options['room_type'] + " — €" +
#     listing_options['price'].astype(str)
# )

# selected_label = st.selectbox("Select a listing", listing_options['label'])
# selected_id = listing_options.loc[listing_options['label'] == selected_label, 'id'].iloc[0]

# # --- Run recommendation ---
# listing_row = get_listing_row(selected_id)
# result = recommend_price(selected_id)
# curve = result['curve']

# actual_price = listing_row['price']
# recommended_price = result['recommended_price']
# expected_revenue = result['expected_revenue_30d']

# # --- Top-line metrics ---
# col1, col2, col3 = st.columns(3)
# col1.metric("Current price", f"€{actual_price:.0f}")
# col2.metric("Recommended price", f"€{recommended_price:.0f}",
#             delta=f"{recommended_price - actual_price:+.0f}")
# col3.metric("Expected 30-day revenue at recommended price", f"€{expected_revenue:,.0f}")

# if result['bound_constrained']:
#     st.warning(
#         "This recommendation sits at the edge of the trusted comp-price range. "
#         "It reflects the boundary of reliable comparable data, not a modeled demand peak — "
#         "treat it as a lower-confidence estimate."
#     )

# # --- Revenue curve chart ---
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=curve['price'], y=curve['expected_revenue_30d'],
#                           mode='lines', name='Raw', line=dict(color='lightgray')))
# fig.add_trace(go.Scatter(x=curve['price'], y=curve['expected_revenue_30d_smoothed'],
#                           mode='lines', name='Smoothed', line=dict(color='steelblue', width=3)))
# fig.add_vline(x=actual_price, line_dash="dash", line_color="red",
#               annotation_text="Current price")
# fig.add_vline(x=recommended_price, line_dash="dash", line_color="green",
#               annotation_text="Recommended")
# fig.update_layout(title="Expected 30-day revenue vs. price",
#                    xaxis_title="Price (€)", yaxis_title="Expected revenue (€)")
# st.plotly_chart(fig, width='stretch')

# # --- Listing details (collapsed by default) ---
# with st.expander("Listing details"):
#     st.json({k: v for k, v in listing_row.items() if k in [
#         'neighbourhood_cleansed', 'room_type', 'accommodates', 'bedrooms',
#         'beds', 'review_scores_rating', 'host_is_superhost', 'local_comp_size', 'citywide_comp_size'
#     ]})
    
# import plotly.express as px

# st.subheader("Click a listing on the map")

# map_df = (
#     full_df.filter(pl.col('snapshot_date') == SNAPSHOT)
#     .select(['id', 'latitude', 'longitude', 'neighbourhood_cleansed', 'room_type', 'price'])
#     .to_pandas()
# )

# fig_map = px.scatter_mapbox(
#     map_df,
#     lat='latitude', lon='longitude',
#     color='price', size_max=8, zoom=11,
#     hover_data=['neighbourhood_cleansed', 'room_type', 'price'],
#     color_continuous_scale='Viridis',
#     mapbox_style='carto-positron',
# )
# fig_map.update_layout(height=550, margin=dict(l=0, r=0, t=0, b=0))

# event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="listing_map")

# if event and event.selection and event.selection.points:
#     point = event.selection.points[0]
#     selected_id = map_df.iloc[point['point_index']]['id']
# else:
#     st.info("Click a point on the map to select a listing.")
#     st.stop()

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulate import (
    full_df, get_listing_row, recommend_price, SNAPSHOT
)

st.set_page_config(page_title="Amsterdam Airbnb Pricing Engine", layout="wide")
st.title("Amsterdam Airbnb Dynamic Pricing")
st.caption("Hedonic pricing + comp-based occupancy modeling — a recommendation, not a causal price effect")

# --- Build the map dataframe once per session ---
@st.cache_data
def build_map_df():
    cols = ['id', 'latitude', 'longitude', 'neighbourhood_cleansed', 'room_type', 'price']
    optional = ['name', 'accommodates', 'bedrooms', 'beds', 'review_scores_rating',
                'host_is_superhost', 'number_of_reviews']
    cols += [c for c in optional if c in full_df.columns]

    df = (
    full_df.filter(pl.col('snapshot_date') == SNAPSHOT)
    .select(cols)
    .drop_nulls(subset=['latitude', 'longitude'])
    .with_columns(pl.col('id').cast(pl.Utf8))   # cast BEFORE to_pandas/Plotly touch it
    .to_pandas()
)
    df['listing_label'] = df['name'] if 'name' in df.columns else (
        df['room_type'] + " in " + df['neighbourhood_cleansed']
    )
    return df

map_df = build_map_df()

# --- Persist selection across reruns ---
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = map_df.iloc[0]['id']  # default to first listing on load

# --- Map ---
st.subheader("Click a listing")

fig_map = px.scatter_map(
    map_df,
    lat='latitude', lon='longitude',
    color='price',
    color_continuous_scale='Viridis',
    zoom=11, height=500,
    map_style='carto-positron',
    hover_name='listing_label',
    hover_data={'price': True, 'room_type': True, 'latitude': False, 'longitude': False},
    custom_data=['id'],  # <-- carries the real id through the click event, no index guessing
)
fig_map.update_traces(marker=dict(size=9))
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))

event = st.plotly_chart(fig_map, width='stretch', on_select="rerun", key="listing_map")

if event and event.selection and event.selection.points:
    clicked_id = event.selection.points[0]['customdata'][0]
    st.session_state.selected_id = clicked_id

selected_id = st.session_state.selected_id

# --- Pull listing info + run recommendation ---
listing_row = get_listing_row(selected_id)
result = recommend_price(selected_id)
curve = result['curve']

actual_price = listing_row['price']
recommended_price = result['recommended_price']
expected_revenue = result['expected_revenue_30d']

st.divider()

# --- Listing identity card ---
left, right = st.columns([2, 1])
with left:
    label = listing_row.get('name') or f"{listing_row['room_type']} in {listing_row['neighbourhood_cleansed'].replace('_', ' ').title()}"
    st.markdown(f"### {label}")
    badges = []
    if listing_row.get('host_is_superhost'):
        badges.append("🌟 Superhost")
    if 'accommodates' in listing_row:
        badges.append(f"👥 Sleeps {int(listing_row['accommodates'])}")
    if 'bedrooms' in listing_row and pd.notna(listing_row['bedrooms']):
        badges.append(f"🛏️ {int(listing_row['bedrooms'])} bedroom(s)")
    if 'review_scores_rating' in listing_row and pd.notna(listing_row['review_scores_rating']):
        badges.append(f"⭐ {listing_row['review_scores_rating']:.2f}")
    st.markdown(" · ".join(badges))
    st.caption(f"{listing_row['neighbourhood_cleansed'].replace('_', ' ').title()} · {listing_row['room_type']}")

with right:
    st.metric("Current price", f"€{actual_price:.0f}/night")

# --- Recommendation metrics ---
col1, col2 = st.columns(2)
col1.metric("Recommended price", f"€{recommended_price:.0f}",
            delta=f"{recommended_price - actual_price:+.0f} vs. current")
col2.metric("Expected 30-day revenue at recommended price", f"€{expected_revenue:,.0f}")

if result['bound_constrained']:
    st.warning(
        "This recommendation sits at the edge of the trusted comp-price range. "
        "It reflects the boundary of reliable comparable data, not a modeled demand peak — "
        "treat it as lower-confidence."
    )

# --- Revenue curve ---
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=curve['price'], y=curve['expected_revenue_30d'],
                                mode='lines', name='Raw', line=dict(color='lightgray')))
fig_curve.add_trace(go.Scatter(x=curve['price'], y=curve['expected_revenue_30d_smoothed'],
                                mode='lines', name='Smoothed', line=dict(color='steelblue', width=3)))
fig_curve.add_vline(x=actual_price, line_dash="dash", line_color="red", annotation_text="Current")
fig_curve.add_vline(x=recommended_price, line_dash="dash", line_color="green", annotation_text="Recommended")
fig_curve.update_layout(title="Expected 30-day revenue vs. price",
                         xaxis_title="Price (€)", yaxis_title="Expected revenue (€)",
                         height=400)
st.plotly_chart(fig_curve, width='stretch')