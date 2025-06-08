# csc-brain
Cambridge Street Capital Backend

This is the research and modeling engine for Cambridge Street Capital.

It handles:
- Historical market data ingestion
- Feature engineering and labeling
- Model training and evaluation
- Backtesting strategies

## Structure

- `data_ingestion/` — data pipeline to fetch and clean raw prices
- `models/` — machine learning strategies (regression, classification)
- `backtester/` — event-driven simulation engine for strategies
