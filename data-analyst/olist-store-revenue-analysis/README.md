# 📊 Brazilian E-Commerce Revenue Analysis Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://olist-store-analysis-project.streamlit.app/) [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

## 🔍 Project Overview

An interactive dashboard analyzing the Brazilian E-Commerce Public Dataset by Olist, providing comprehensive insights into sales trends, product performance, customer demographics, and customer value through RFM analysis.

![Dashboard Preview](https://img.freepik.com/free-vector/ecommerce-web-page-concept-illustration_114360-8204.jpg?w=500)

## 📈 Key Metrics & Insights

### 1. Sales Performance
- **Total Orders**: Track the cumulative number of orders over time
- **Total Revenue**: Monitor the overall revenue in Brazilian Real (BRL)
- **Daily Revenue Trends**: Visualize day-by-day revenue fluctuations with interactive time series graph

### 2. Product Analysis
- **Top 10 Best-Selling Categories**: Identify the most popular product categories by order count
- **Bottom 10 Underperforming Categories**: Pinpoint product categories with lowest sales volume

### 3. Customer Demographics
- **Geographic Distribution**: View top 10 cities by order count
- **Regional Sales Concentration**: Identify key market areas for targeted marketing

### 4. RFM Customer Segmentation
- **Recency**: Identify customers who purchased most recently
- **Frequency**: Highlight customers with highest purchase frequency
- **Monetary Value**: Showcase top customers by spending amount

## 🚀 Live Dashboard

Explore the interactive dashboard: [Olist Store Analysis Dashboard](https://olist-store-analysis-project.streamlit.app/)

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Environment Setup
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Running the Dashboard Locally
```bash
streamlit run dashboard.py
```

## 📝 Business Questions Addressed

1. How have sales fluctuated over time?
2. What are the most and least sold product categories?
3. Where are most customers ordering from?
4. Who are our most valuable customers based on RFM analysis?

## 🔗 Data Source

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) available on Kaggle, containing information on 100k orders from 2016 to 2018.

