from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'

st.set_page_config(page_title='Retail Sales Intelligence & Forecasting System', page_icon='📊', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / 'cleaned_superstore_final.csv', parse_dates=['Order Date', 'Ship Date'])
    monthly = pd.read_csv(OUT_DIR / 'monthly_sales_series.csv', parse_dates=['Date'])
    forecast = pd.read_csv(OUT_DIR / 'future_6_month_forecast.csv', parse_dates=['Date'])
    performance = pd.read_csv(OUT_DIR / 'model_performance.csv')
    fi = pd.read_csv(OUT_DIR / 'feature_importance.csv')
    with open(OUT_DIR / 'project_summary.json') as f:
        summary = json.load(f)
    return df, monthly, forecast, performance, fi, summary


df, monthly, forecast, performance, fi, summary = load_data()

st.title('Retail Sales Intelligence & Forecasting System')
st.caption('Business Intelligence final project built with Python, Scikit-learn, and Streamlit')

st.sidebar.header('Filters')
regions = st.sidebar.multiselect('Region', sorted(df['Region'].unique()), default=sorted(df['Region'].unique()))
categories = st.sidebar.multiselect('Category', sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))
years = st.sidebar.multiselect('Year', sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))

filtered = df[df['Region'].isin(regions) & df['Category'].isin(categories) & df['Year'].isin(years)].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Sales', f"${filtered['Sales'].sum():,.0f}")
col2.metric('Total Profit', f"${filtered['Profit'].sum():,.0f}")
col3.metric('Average Order Value', f"${filtered['Sales'].mean():,.2f}")
col4.metric('Best Forecast Model (RMSE & R²)', summary['best_model'])

overview_tab, dashboard_tab, forecast_tab = st.tabs(['Overview', 'BI Dashboard', 'Forecasting'])

with overview_tab:
    st.subheader('Project Summary')
    st.write(
        'This application supports retail decision-making by combining descriptive analytics with '
        'monthly sales forecasting. It helps managers understand product, region, and discount performance '
        'while also estimating future sales trends.'
    )
    st.write(
        'This system enables data-driven decision-making by integrating data preprocessing, exploratory analysis, '
        'machine learning, and interactive dashboarding in one end-to-end Business Intelligence solution.'
    )
    st.markdown(
        f"""
        - **Dataset:** Superstore retail transactions ({summary['dataset_rows']:,} rows, {summary['date_start']} to {summary['date_end']})
        - **Top category:** {summary['top_category']} (${summary['top_category_sales']:,.0f})
        - **Top region:** {summary['top_region']} (${summary['top_region_sales']:,.0f})
        - **Correlation (Sales vs Profit):** {summary['sales_profit_correlation']:.3f}
        - **Correlation (Discount vs Profit):** {summary['discount_profit_correlation']:.3f}
        - **Best forecast model:** {summary['best_model']} (RMSE: {summary['best_model_rmse']:,.0f}, R²: {summary['best_model_r2']:.4f})
        """
    )
    st.dataframe(performance, use_container_width=True, hide_index=True)

with dashboard_tab:
    st.subheader('Interactive Business Intelligence Dashboard')
    left, right = st.columns(2)
    cat_sales = filtered.groupby('Category', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
    fig_cat = px.bar(cat_sales, x='Category', y='Sales', title='Sales by Category', text_auto='.2s')
    left.plotly_chart(fig_cat, use_container_width=True)

    reg_sales = filtered.groupby('Region', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
    fig_reg = px.bar(reg_sales, x='Region', y='Sales', title='Sales by Region', text_auto='.2s')
    right.plotly_chart(fig_reg, use_container_width=True)

    monthly_filtered = filtered.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    fig_monthly = px.line(monthly_filtered, x='Order Date', y='Sales', title='Monthly Sales Trend', markers=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

    left2, right2 = st.columns(2)
    fig_scatter = px.scatter(
        filtered.sample(min(3000, len(filtered)), random_state=42),
        x='Discount', y='Profit', color='Category',
        title='Discount vs Profit'
    )
    left2.plotly_chart(fig_scatter, use_container_width=True)

    top_products = filtered.groupby('Product Name', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False).head(10)
    fig_prod = px.bar(top_products.sort_values('Sales'), x='Sales', y='Product Name', orientation='h', title='Top 10 Products by Sales')
    right2.plotly_chart(fig_prod, use_container_width=True)

with forecast_tab:
    st.subheader('Monthly Sales Forecasting')
    st.write(
    'This section compares model performance, evaluates test predictions, and presents a 6-month sales forecast.'
)
    test_pred = pd.read_csv(OUT_DIR / 'forecast_test_predictions.csv', parse_dates=['Date'])
    fig_eval = px.line(test_pred.melt(id_vars='Date', value_vars=['Actual Sales', 'Predicted Sales']),
                       x='Date', y='value', color='variable', markers=True,
                       title='Actual vs Predicted Sales on Test Set')
    st.plotly_chart(fig_eval, use_container_width=True)

    fig_future = px.line(title='Historical Sales and 6-Month Forecast')
    fig_future.add_scatter(x=monthly['Date'], y=monthly['Monthly Sales'], mode='lines+markers', name='Historical Sales')
    fig_future.add_scatter(x=forecast['Date'], y=forecast['Forecast Sales'], mode='lines+markers', name='Forecast', line=dict(dash='dash'))
    st.plotly_chart(fig_future, use_container_width=True)

    fi_chart = px.bar(fi.head(10).sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Top Forecast Features')
    st.plotly_chart(fi_chart, use_container_width=True)

    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.info('Run the app locally with: `streamlit run app.py`')
