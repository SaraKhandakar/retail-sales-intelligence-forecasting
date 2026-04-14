# 📊 Retail Sales Intelligence & Forecasting System

##  Live Demo
https://retail-sales-intelligence-forecasting-jsyuupracbtsfqffmugm3w.streamlit.app/

---

##  Project Overview
This project is an end-to-end Business Intelligence system built on the Superstore dataset. It combines data preprocessing, exploratory data analysis, business insight generation, machine learning-based forecasting, and an interactive Streamlit dashboard to support data-driven decision-making.

---

##  Problem Statement
Retail businesses often struggle to understand sales patterns, optimize profitability, and accurately forecast future demand. This project addresses these challenges by providing actionable insights and predictive analytics to improve decision-making.

---

##  Project Deliverables
- Data preparation and transformation pipeline
- Exploratory charts and KPI summaries
- Monthly sales forecasting model using Scikit-learn
- Streamlit dashboard for interactive analysis
- IEEE-style final report
- Final PowerPoint presentation

---

##  Project Structure
- `data/raw_superstore.csv` - original dataset  
- `data/cleaned_superstore_final.csv` - processed dataset  
- `src/build_project_assets.py` - reproducible data preparation, analysis, and forecasting script  
- `models/random_forest_monthly_sales.joblib` - trained forecasting model  
- `outputs/` - generated charts, tables, predictions, and summary files  
- `app.py` - Streamlit dashboard  
- `requirements.txt` - Python dependencies  

---

##  Technologies Used
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn, Plotly)
- Machine Learning (Scikit-learn)
- Streamlit (Interactive Dashboard)
- GitHub (Version Control)

---

## ▶ How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


## Launch the dashboard:

- streamlit run app.py

## Forecasting Approach

- The predictive component aggregates transaction data into monthly sales totals and engineers time-series features such as:

- Lag features (1, 2, 3, 6, 12 months)
- Rolling averages (3-month, 6-month)
- Rolling standard deviation
- Year-over-year growth
- Time-based features (month, quarter, year)

- Two models were tested:

   - Linear Regression
   - Random Forest Regressor

- The Random Forest model performed best and was selected for the final system.

## Key Results & Insights
- Technology category generates the highest sales
- West region shows the strongest performance
- Discounts negatively impact profitability
- High sales do not always guarantee high profit
- The best forecasting model achieved:
   - RMSE ≈ 13.9K
   - R² ≈ 0.71

## Error Handling and Logging
- This project includes error handling and logging to improve system reliability.

- The Python logging module is used to record key steps such as data processing, model training, and forecasting. Errors are captured using try–except blocks and saved in log files (`logs/project.log` and `logs/app.log`).

- This helps with debugging, monitoring, and makes the system more robust for real-world use.



## System Architecture

- End-to-End BI Pipeline:

   - Data Collection
   - Data Cleaning & Preprocessing
   - Exploratory Data Analysis (EDA)
   - Feature Engineering
   - Machine Learning Modeling
   - Dashboard Development (Streamlit)
   - Deployment

## Conclusion

- This project demonstrates a complete Business Intelligence pipeline from raw data to deployed application, integrating analytics and machine learning into a real-world decision support system.

## Author

- Shara Khandakar
- CST2213 – Business Intelligence Programming