# Retail Sales Intelligence and Forecasting System using Python

This project is a complete Business Intelligence solution built on the Superstore dataset. It combines data cleaning, exploratory analysis, business insight generation, and monthly sales forecasting in one reproducible Python workflow.

## Project Deliverables
- Data preparation and transformation pipeline
- Exploratory charts and KPI summaries
- Monthly sales forecasting model using Scikit-learn
- Streamlit dashboard for interactive analysis
- IEEE-style final report
- Final PowerPoint presentation

## Project Structure
- `data/raw_superstore.csv` - original dataset
- `data/cleaned_superstore_final.csv` - processed dataset
- `src/build_project_assets.py` - reproducible data preparation, analysis, and forecasting script
- `models/random_forest_monthly_sales.joblib` - trained forecasting model
- `outputs/` - generated charts, tables, predictions, and summary files
- `app.py` - Streamlit dashboard
- `requirements.txt` - Python dependencies

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Rebuild project assets:
   ```bash
   python src/build_project_assets.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Forecasting Approach
The predictive component aggregates transaction data into monthly sales totals and engineers lag-based time-series features such as:
- Lag 1, Lag 2, Lag 3, Lag 6, and Lag 12 months
- 3-month and 6-month rolling averages
- 3-month rolling standard deviation
- year-over-year growth
- time trend, month, quarter, and year

Two models were tested:
- Linear Regression
- Random Forest Regressor

The Random Forest model performed best and was selected for the final system.

## Key Results
- Top sales category: Technology
- Top sales region: West
- Discounts are negatively associated with profit
- The best forecasting model achieved an RMSE of about 13.9K and R² of about 0.71 on the hold-out test set

## Notes
This project is designed to satisfy the Business Intelligence final project requirements for CST2213, including advanced Python, predictive analytics, visualization, documentation, and presentation.
