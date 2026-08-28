# Amsterdam Airbnb Dynamic Pricing & Revenue Optimization

Check it out here: [emads-airbnb-tool.streamlit.app](https://emads-airbnb-tool.streamlit.app/)

A decision-support tool for Airbnb hosts: given a listing, what should it charge, and what's the revenue upside from adjusting price? Built as an end-to-end pipeline — data ingestion, two predictive models, a revenue simulation engine, and an interactive dashboard — using real Amsterdam market data.

This is **not** a demand-curve prediction exercise. It's framed the way commercial pricing tools (PriceLabs, Beyond Pricing, Wheelhouse) actually operate: combine a hedonic "fair value" estimate with comp-based occupancy signal, then simulate revenue outcomes across a range of candidate prices to recommend where a host should sit.

## Why this framing, not classic price elasticity

Airbnb calendar data can't cleanly separate "date was booked" from "host manually blocked the date," so it isn't a clean causal elasticity setup. Rather than overreach into causal language the data can't support, this project makes an honest, narrower claim: **hedonic pricing + comparable-group occupancy modeling.** That's a more defensible claim, and it's also closer to how real pricing tools work in practice.

## How it works

**1. Hedonic pricing model** — predicts a listing's "fair" price from its features: location, room type, capacity, amenities, host quality signals, and review history.

- XGBoost regression on log-price (R² ≈ 0.67), selected over a Ridge baseline for performance
- Feature engineering includes amenities parsing, categorical encoding with deliberate reference baselines, a missing-flag pattern for imputed fields (placeholder value + binary indicator, so the model can distinguish "confirmed absent" from "unknown"), and review-date-derived features

**2. Occupancy / demand model** — estimates relative booking demand using `reviews_per_month` as a demand proxy (calendar-based availability was tested as a proxy and abandoned after showing near-zero predictive signal).

- XGBoost regression (R² ≈ 0.42)
- Built on a comp-group ranking architecture: each listing is compared against others in the same neighbourhood + room type + accommodates bucket, falling back to a city-wide comparison group when the local group is too thin
- Produces `value_gap`, `local_percentile_rank`, and `citywide_percentile_rank` features — how this listing's price compares to what similar listings charge
- Hotel-room listings are excluded from training, since they don't compete in the same market segment as entire homes / private rooms

**3. Simulation & recommendation layer** — combines both models into an actionable output.

- `simulate_revenue_curve` sweeps a range of candidate prices for a listing, recomputing price-dependent comp features at each step against a frozen comp-group distribution (so the simulation doesn't quietly drift the comparison set as price changes)
- `build_comp_cache` precomputes and persists comp-group distributions so simulation doesn't recompute them per listing
- `get_price_bounds` caps the sweep range at empirical percentiles, avoiding nonsensical extrapolation
- `recommend_price` returns a recommended price along with a `bound_constrained` flag, distinguishing a genuine interior revenue-maximizing price from one that's just sitting at the edge of the allowed sweep range (an artifact, not a real optimum)

**4. Dashboard** — a Streamlit app for exploring the tool interactively:

- Clickable map of listings
- Listing identity card (features, current price, comp group)
- Recommended price and expected revenue impact
- Revenue curve chart showing simulated revenue across the price sweep

## Data

Source: [Inside Airbnb](http://insideairbnb.com/) — Amsterdam.

Two quarterly snapshots are used:

- `sep-11-2025/` — listings, calendar, and reviews
- `jun-15-2026/` — listings, calendar, and reviews

Two snapshots give a lightweight before/after signal but are **not** a substitute for a real panel — with only two confounded time points, seasonal effects and reverse causality can't be cleanly separated from genuine price effects. The pooled cross-section is used for modeling; the two-snapshot comparison is corroborative context only, not causal evidence.

Raw CSVs are cleaned and cached as Parquet (`data/concat_listings.parquet`, `data/concat_calenders.parquet`, `data/merged_listings.parquet`, etc.) to avoid re-parsing on every run. Comp-group lookups are cached separately (`data/local_groups.joblib`, `data/citywide_groups.joblib`).

## Repo structure

```
airbnb/
├── src/
│   ├── cleaning.py            # Raw CSV ingestion, cleaning, merging across snapshots
│   ├── features_hedonic.py    # Feature engineering for the hedonic price model
│   ├── features_occupancy.py  # Comp-group construction, demand-proxy features
│   ├── model_hedonic.py       # Hedonic pricing model: training, SHAP, evaluation
│   ├── model_occupancy.py     # Occupancy/demand model: training, evaluation
│   ├── simulate.py            # Revenue curve simulation + price recommendation logic
│   └── dashboard.py           # Streamlit app
├── notebooks/                 # Exploratory work: EDA, feature iteration, model tuning
├── models/                    # Serialized XGBoost models (joblib)
├── data/                      # Cleaned Parquet datasets, comp-group caches, raw snapshots
├── figs/                      # Exploratory plots (price distributions, seasonal patterns, etc.)
├── pyproject.toml / uv.lock   # Dependencies (managed with uv)
└── .env                       # Local data path configuration (not committed)
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone <repo-url>
cd airbnb
uv sync
```

Create a `.env` file in the project root pointing to your local data paths:

```
CONCAT_LISTINGS_PATH=data/concat_listings.parquet
CONCAT_CALENDER_PATH=data/concat_calenders.parquet
DATA_PATH=data/
PRED_PATH=data/hedonic_predictions.parquet
MODEL_PATH=models/
```

Run the dashboard:

```bash
uv run streamlit run src/dashboard.py
```

## Tech stack

Python · Polars (primary data manipulation) · XGBoost · scikit-learn · Streamlit · Plotly (`scatter_map`) · joblib · uv

## Known limitations

- **Not causal.** This tool estimates fair value and relative demand from cross-sectional and comp-based signal — it does not claim to isolate the causal effect of price on bookings. See framing note above.
- **Backtested revenue-lift figures are not yet included.** An initial backtest pipeline exists (`simulate.py`), but current output numbers are inflated due to a placeholder conversion factor and weak learned price elasticity in the occupancy model. This will be corrected and added once validated.
- **Large-capacity boat listings** are systematically over/under-predicted by the hedonic model, due to a missing `accommodates × dwelling_type` interaction term — a documented issue for future iteration, not currently corrected for.
- Two quarterly snapshots provide useful corroborative signal but are not a full panel; results should not be read as tracking true within-listing price effects over time.

## Roadmap

- Correct the backtest conversion factor and re-validate revenue-lift claims
- Add the `accommodates × dwelling_type` interaction to the hedonic model
- Package into a cleanly reusable module structure to support ingesting a second city's data as a generalizability demo
