from __future__ import annotations
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'
MODEL_DIR = BASE / 'models'
LOG_DIR = BASE / 'logs'
LOG_FILE = LOG_DIR / 'project.log'

for d in [DATA_DIR, OUT_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'

RAW_FILE = DATA_DIR / 'raw_superstore.csv'


def load_and_prepare_data() -> pd.DataFrame:
    try:
        logging.info("Loading raw dataset from %s", RAW_FILE)
        df = pd.read_csv(RAW_FILE, encoding='latin1')

        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month_name()
        df['MonthNum'] = df['Order Date'].dt.month
        df['Profit Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], 0)
        df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days

        logging.info("Data loaded and prepared successfully. Shape: %s", df.shape)
        return df
    except Exception as e:
        logging.error("Error in load_and_prepare_data: %s", str(e), exc_info=True)
        raise


def create_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Creating monthly sales series")
        monthly = (
            df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales']
            .sum()
            .reset_index()
            .rename(columns={'Order Date': 'Date', 'Sales': 'Monthly Sales'})
        )

        monthly['Year'] = monthly['Date'].dt.year
        monthly['Month'] = monthly['Date'].dt.month
        monthly['Quarter'] = monthly['Date'].dt.quarter
        monthly['Trend'] = np.arange(len(monthly))

        for lag in [1, 2, 3, 6, 12]:
            monthly[f'Lag_{lag}'] = monthly['Monthly Sales'].shift(lag)

        monthly['RollingMean_3'] = monthly['Monthly Sales'].shift(1).rolling(3).mean()
        monthly['RollingStd_3'] = monthly['Monthly Sales'].shift(1).rolling(3).std()
        monthly['RollingMean_6'] = monthly['Monthly Sales'].shift(1).rolling(6).mean()
        monthly['YoY_Growth'] = monthly['Monthly Sales'].pct_change(12)

        logging.info("Monthly series created successfully. Shape: %s", monthly.shape)
        return monthly
    except Exception as e:
        logging.error("Error in create_monthly_series: %s", str(e), exc_info=True)
        raise


def evaluate_forecast_models(monthly_features: pd.DataFrame):
    try:
        logging.info("Evaluating forecasting models")
        modeling = monthly_features.dropna().copy()

        feature_cols = [
            'Year', 'Month', 'Quarter', 'Trend',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_6', 'Lag_12',
            'RollingMean_3', 'RollingStd_3', 'RollingMean_6', 'YoY_Growth'
        ]

        X = modeling[feature_cols]
        y = modeling['Monthly Sales']

        split_idx = len(modeling) - 12
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        d_test = modeling['Date'].iloc[split_idx:]

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                max_depth=6,
                min_samples_split=2,
                min_samples_leaf=1,
            ),
        }

        performance_rows = []
        prediction_frames = {}
        fitted = {}

        for name, model in models.items():
            logging.info("Training model: %s", name)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

            performance_rows.append({
                'Model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R2': round(r2, 4),
                'MAPE (%)': round(mape, 2),
            })

            prediction_frames[name] = pd.DataFrame({
                'Date': d_test,
                'Actual Sales': y_test.values,
                'Predicted Sales': pred,
                'Residual': y_test.values - pred,
            })

            fitted[name] = model
            logging.info(
                "Completed model: %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f",
                name, mae, rmse, r2, mape
            )

        performance = pd.DataFrame(performance_rows).sort_values(['RMSE', 'MAE']).reset_index(drop=True)
        best_model_name = performance.iloc[0]['Model']
        best_model = fitted[best_model_name]

        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': getattr(best_model, 'feature_importances_', np.repeat(0, len(feature_cols)))
        }).sort_values('Importance', ascending=False)

        logging.info("Best model selected: %s", best_model_name)

        return {
            'feature_cols': feature_cols,
            'modeling_df': modeling,
            'performance': performance,
            'predictions': prediction_frames,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'feature_importance': feature_importance,
        }
    except Exception as e:
        logging.error("Error in evaluate_forecast_models: %s", str(e), exc_info=True)
        raise


def recursive_forecast(monthly_all: pd.DataFrame, trained_model, feature_cols, steps: int = 6) -> pd.DataFrame:
    try:
        logging.info("Starting recursive %s-month forecast", steps)
        hist = monthly_all.copy().sort_values('Date').reset_index(drop=True)
        preds = []

        for step in range(steps):
            next_date = hist['Date'].iloc[-1] + pd.offsets.MonthBegin(1)
            row = {
                'Date': next_date,
                'Year': next_date.year,
                'Month': next_date.month,
                'Quarter': pd.Timestamp(next_date).quarter,
                'Trend': hist['Trend'].iloc[-1] + 1,
            }

            sales_series = hist['Monthly Sales']
            for lag in [1, 2, 3, 6, 12]:
                row[f'Lag_{lag}'] = sales_series.iloc[-lag]

            row['RollingMean_3'] = sales_series.iloc[-3:].mean()
            row['RollingStd_3'] = sales_series.iloc[-3:].std()
            row['RollingMean_6'] = sales_series.iloc[-6:].mean()
            row['YoY_Growth'] = (sales_series.iloc[-1] / sales_series.iloc[-12]) - 1 if len(sales_series) >= 12 else 0

            pred = float(trained_model.predict(pd.DataFrame([row])[feature_cols])[0])
            row['Monthly Sales'] = pred

            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
            preds.append({'Date': next_date, 'Forecast Sales': round(pred, 2)})

            logging.info("Forecast step %s completed for %s: %.2f", step + 1, next_date.date(), pred)

        return pd.DataFrame(preds)
    except Exception as e:
        logging.error("Error in recursive_forecast: %s", str(e), exc_info=True)
        raise


def save_visualizations(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    perf: pd.DataFrame,
    preds: pd.DataFrame,
    fi: pd.DataFrame,
    future_fc: pd.DataFrame
):
    try:
        logging.info("Saving visualizations")

        cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(8, 4.6))
        ax = sns.barplot(x=cat_sales.index, y=cat_sales.values, hue=cat_sales.index, dodge=False, legend=False)
        ax.set_title('Total Sales by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Sales (USD)')
        for i, v in enumerate(cat_sales.values):
            ax.text(i, v + cat_sales.max() * 0.015, f'${v/1000:.0f}K', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'category_sales.png')
        plt.close()

        reg_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        plt.figure(figsize=(8, 4.6))
        ax = sns.barplot(x=reg_sales.index, y=reg_sales.values, hue=reg_sales.index, dodge=False, legend=False)
        ax.set_title('Total Sales by Region')
        ax.set_xlabel('Region')
        ax.set_ylabel('Sales (USD)')
        for i, v in enumerate(reg_sales.values):
            ax.text(i, v + reg_sales.max() * 0.015, f'${v/1000:.0f}K', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'region_sales.png')
        plt.close()

        plt.figure(figsize=(9, 4.8))
        ax = sns.lineplot(data=monthly, x='Date', y='Monthly Sales', marker='o')
        ax.set_title('Monthly Sales Trend (2014-2017)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'monthly_sales_trend.png')
        plt.close()

        sample_df = df.sample(min(len(df), 2500), random_state=42)
        plt.figure(figsize=(8, 4.8))
        ax = sns.scatterplot(data=sample_df, x='Discount', y='Profit', alpha=0.5)
        sns.regplot(data=sample_df, x='Discount', y='Profit', scatter=False, ax=ax, truncate=False)
        ax.set_title('Discount vs Profit')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'discount_vs_profit.png')
        plt.close()

        corr = df[['Sales', 'Profit', 'Quantity', 'Discount', 'Profit Margin', 'Shipping Days']].corr()
        plt.figure(figsize=(6.5, 5.2))
        sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', square=True)
        plt.title('Correlation Matrix of Key Numeric Features')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'correlation_heatmap.png')
        plt.close()

        plt.figure(figsize=(9, 4.8))
        plt.plot(preds['Date'], preds['Actual Sales'], marker='o', label='Actual Sales')
        plt.plot(preds['Date'], preds['Predicted Sales'], marker='o', label='Predicted Sales')
        plt.title('Random Forest Monthly Forecast: Actual vs Predicted')
        plt.xlabel('Month')
        plt.ylabel('Sales (USD)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'actual_vs_predicted.png')
        plt.close()

        plt.figure(figsize=(8, 5.5))
        top_fi = fi.head(10).sort_values('Importance')
        plt.barh(top_fi['Feature'], top_fi['Importance'])
        plt.title('Top Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'feature_importance.png')
        plt.close()

        plt.figure(figsize=(9, 4.8))
        hist_tail = monthly.tail(18)
        plt.plot(hist_tail['Date'], hist_tail['Monthly Sales'], marker='o', label='Historical Sales')
        plt.plot(future_fc['Date'], future_fc['Forecast Sales'], marker='o', linestyle='--', label='6-Month Forecast')
        plt.title('Historical Monthly Sales and 6-Month Forecast')
        plt.xlabel('Month')
        plt.ylabel('Sales (USD)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'future_forecast.png')
        plt.close()

        top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(9, 5.2))
        plt.barh(top_products.index[::-1], top_products.values[::-1])
        plt.title('Top 10 Products by Total Sales')
        plt.xlabel('Sales (USD)')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'top_products.png')
        plt.close()

        logging.info("Visualizations saved successfully")
    except Exception as e:
        logging.error("Error in save_visualizations: %s", str(e), exc_info=True)
        raise


def build_summary(df: pd.DataFrame, monthly: pd.DataFrame, performance: pd.DataFrame, future_fc: pd.DataFrame) -> dict:
    try:
        logging.info("Building project summary")
        top_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        top_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        corr_sales_profit = df['Sales'].corr(df['Profit'])
        corr_discount_profit = df['Discount'].corr(df['Profit'])
        monthly_best = monthly.loc[monthly['Monthly Sales'].idxmax()]
        monthly_low = monthly.loc[monthly['Monthly Sales'].idxmin()]

        summary = {
            'dataset_rows': int(df.shape[0]),
            'dataset_columns': int(df.shape[1]),
            'date_start': str(df['Order Date'].min().date()),
            'date_end': str(df['Order Date'].max().date()),
            'total_sales': round(float(df['Sales'].sum()), 2),
            'total_profit': round(float(df['Profit'].sum()), 2),
            'avg_order_value': round(float(df['Sales'].mean()), 2),
            'top_category': top_category.index[0],
            'top_category_sales': round(float(top_category.iloc[0]), 2),
            'top_region': top_region.index[0],
            'top_region_sales': round(float(top_region.iloc[0]), 2),
            'sales_profit_correlation': round(float(corr_sales_profit), 3),
            'discount_profit_correlation': round(float(corr_discount_profit), 3),
            'best_month': str(monthly_best['Date'].date()),
            'best_month_sales': round(float(monthly_best['Monthly Sales']), 2),
            'lowest_month': str(monthly_low['Date'].date()),
            'lowest_month_sales': round(float(monthly_low['Monthly Sales']), 2),
            'best_model': performance.iloc[0]['Model'],
            'best_model_rmse': float(performance.iloc[0]['RMSE']),
            'best_model_mae': float(performance.iloc[0]['MAE']),
            'best_model_r2': float(performance.iloc[0]['R2']),
            'forecast_next_month': round(float(future_fc.iloc[0]['Forecast Sales']), 2),
            'forecast_horizon_avg': round(float(future_fc['Forecast Sales'].mean()), 2),
        }

        logging.info("Project summary built successfully")
        return summary
    except Exception as e:
        logging.error("Error in build_summary: %s", str(e), exc_info=True)
        raise


def save_tables(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    performance: pd.DataFrame,
    preds: pd.DataFrame,
    future_fc: pd.DataFrame
):
    try:
        logging.info("Saving output tables")
        category_table = df.groupby('Category').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).round(2).sort_values('Total_Sales', ascending=False)

        region_table = df.groupby('Region').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).round(2).sort_values('Total_Sales', ascending=False)

        top_products = df.groupby('Product Name').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).round(2).sort_values('Total_Sales', ascending=False).head(10)

        category_table.to_csv(OUT_DIR / 'category_summary.csv')
        region_table.to_csv(OUT_DIR / 'region_summary.csv')
        performance.to_csv(OUT_DIR / 'model_performance.csv', index=False)
        preds.to_csv(OUT_DIR / 'forecast_test_predictions.csv', index=False)
        future_fc.to_csv(OUT_DIR / 'future_6_month_forecast.csv', index=False)
        top_products.to_csv(OUT_DIR / 'top_products_summary.csv')
        monthly.to_csv(OUT_DIR / 'monthly_sales_series.csv', index=False)

        logging.info("Output tables saved successfully")
    except Exception as e:
        logging.error("Error in save_tables: %s", str(e), exc_info=True)
        raise


def main():
    try:
        logging.info("Starting BI project asset generation pipeline")

        df = load_and_prepare_data()
        df.to_csv(DATA_DIR / 'cleaned_superstore_final.csv', index=False)
        logging.info("Cleaned dataset saved")

        monthly = create_monthly_series(df)

        evaluation = evaluate_forecast_models(monthly)
        performance = evaluation['performance']
        best_model_name = evaluation['best_model_name']
        best_model = evaluation['best_model']
        preds = evaluation['predictions'][best_model_name]
        fi = evaluation['feature_importance']

        forecast_6 = recursive_forecast(monthly, best_model, evaluation['feature_cols'], 6)

        save_tables(df, monthly, performance, preds, forecast_6)
        save_visualizations(df, monthly, performance, preds, fi, forecast_6)

        summary = build_summary(df, monthly, performance, forecast_6)

        with open(OUT_DIR / 'project_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        fi.to_csv(OUT_DIR / 'feature_importance.csv', index=False)
        joblib.dump(best_model, MODEL_DIR / 'random_forest_monthly_sales.joblib')

        logging.info("Feature importance saved to outputs/feature_importance.csv")
        logging.info("Best model saved to models/random_forest_monthly_sales.joblib")
        logging.info("Pipeline completed successfully")

        print('Best model:', best_model_name)
        print(performance.to_string(index=False))
        print('Forecast next 6 months:\n', forecast_6.to_string(index=False))

    except Exception as e:
        logging.error("Pipeline failed in main(): %s", str(e), exc_info=True)
        print("An error occurred. Check logs/project.log for details.")
        raise


if __name__ == '__main__':
    main()