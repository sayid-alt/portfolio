import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from babel.numbers import format_currency

sns.set(style='dark')

# create daily orders report


def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        'order_id': "nunique",
        'price': 'sum'
    }).reset_index()

    daily_orders_df.rename(columns={
        'order_purchase_timestamp': 'date',
        'order_id': 'order_count',
        'price': 'revenue'
    }, inplace=True)

    return daily_orders_df


# create most sold product report
def create_most_product_order_df(df):
    # create sample datasets of particular range time
    most_product_orders_df = df.groupby(by=['product_category_name_english']).agg({
        'order_id': 'nunique'
    }).sort_values(by='order_id', ascending=False).reset_index()

    most_product_orders_df.rename(columns={
        'product_category_name_english': 'category',
        'order_id': 'category_count'
    }, inplace=True)

    return most_product_orders_df


def create_order_bycity_df(df):
    order_bycity_df = df.groupby(by='customer_city').agg({
        'order_id': 'nunique',
        'price': 'sum'
    }).sort_values(by='order_id', ascending=False).reset_index()

    order_bycity_df.rename(columns={
        'customer_city': 'city',
        'order_id': 'order_count',
        'price': 'revenue'
    }, inplace=True)

    return order_bycity_df


def create_rfm_df(df):
    rfm_df = df.groupby('customer_id', as_index=False).agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'price': 'sum'
    })
    rfm_df.columns = ['customer_id',
                      'max_order_timestamp', 'frequency', 'monetary']

    recent_date = df['order_purchase_timestamp'].dt.date.max()
    rfm_df['max_order_timestamp'] = rfm_df['max_order_timestamp'].dt.date

    # calculate recent date of customer order
    rfm_df['recency'] = rfm_df['max_order_timestamp'].apply(
        lambda x: (recent_date - x).days)
    rfm_df['customer_number'] = [x+1 for x in range(len(rfm_df))]

    return rfm_df


# ----------------------------------- main code ------------------------------------------
# load dataset
all_df = pd.read_csv('./data/all_data.csv')

# initialize the datetime column
datetime_columns = ['order_purchase_timestamp', 'order_approved_at',
                    'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date']


# sorting dataset by order datetime
all_df.sort_values(by='order_purchase_timestamp', inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# initialize min and max order date
min_date = all_df['order_purchase_timestamp'].dt.date.min()
max_date = all_df['order_purchase_timestamp'].dt.date.max()


with st.sidebar:
    st.image('https://img.freepik.com/free-vector/ecommerce-web-page-concept-illustration_114360-8204.jpg?t=st=1709652298~exp=1709655898~hmac=6f17130b71c17650efea218f83134e1dd803c8d1b4f49ed22018b04d4529d2ae&w=2000')
    # initialize start_date and end date of the range
    start_date, end_date = st.date_input(
        'Input date range: ({} - {})'.format(min_date, max_date),
        min_value=min_date, max_value=max_date,
        value=[min_date, max_date]
    )


# create filtered dataset as a range valid date
main_df = all_df[(all_df['order_purchase_timestamp'] >= str(start_date)) & (
    all_df['order_purchase_timestamp'] <= str(end_date))]


# initialize attribute visualizaition datasets
daily_orders_df = create_daily_orders_df(main_df)
most_product_orders_df = create_most_product_order_df(main_df)
order_bycity_df = create_order_bycity_df(main_df)
rfm_df = create_rfm_df(main_df)


st.title('Dashboard Brazilian E-Commerce Public Dataset')

# daily report dashboard
st.subheader('Daily Report')
col1, col2 = st.columns(2)
# matric total orders
with col1:
    total_orders = daily_orders_df['order_count'].sum()
    st.metric(label='Total Orders', value=total_orders)

with col2:
    total_revenue = format_currency(
        daily_orders_df['revenue'].sum(), 'BRL', locale='pt_BR')
    st.metric(label='Total Revenue', value=total_revenue)

# Show graph of daily order
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(daily_orders_df['date'],
        daily_orders_df['revenue'], linewidth=1, color='#863A6F')

plt.xticks(rotation=45)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)


# Most sold product report
st.subheader('Most Ordered Product')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

colors = ['#863A6F', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3',
          '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3']

sns.barplot(
    data=most_product_orders_df.sort_values(
        by='category_count', ascending=False).head(10),
    y='category', x='category_count',
    palette=colors,
    ax=ax[0]
)

ax[0].set_title('The most sold product', fontsize=20)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].tick_params('y', labelsize=20)
ax[0].tick_params('x', labelsize=15)


sns.barplot(
    data=most_product_orders_df.sort_values(
        by='category_count', ascending=True).tail(10),
    y='category', x='category_count',
    palette=colors,
    ax=ax[1]
)

ax[1].set_title('The least sold product', fontsize=20)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].yaxis.set_label_position('right')  # set label y axis to the right
ax[1].tick_params(axis='y', labelsize=20)
ax[1].tick_params(axis='x', labelsize=15)

# invert xaxis direction
ax[1].invert_xaxis()

# set yaxis label to the right side
ax[1].yaxis.tick_right()

st.pyplot(fig)

# number of Order by city
st.subheader('Customer Demographichs')
colors = ['#863A6F', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3',
          '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3', '#D3D3D3']

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(
    data=order_bycity_df.sort_values(
        by='order_count', ascending=False).head(10),
    x='city', y='order_count',
    palette=colors
)

ax.set_ylabel(None)
ax.set_ylabel(None)
ax.set_title('Number of order by city', fontsize=15)
plt.xticks(rotation=45)
st.pyplot(fig)


# report RFM analysis
st.subheader('Best Customer based on RFM analysis')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))

colors = ['#FFADBC', '#FFADBC', '#FFADBC', '#FFADBC', '#FFADBC', '#FFADBC']


sns.barplot(
    y="recency", x="customer_number",
    data=rfm_df.sort_values(by="recency").head(),
    palette=colors,
    ax=ax[0])

ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Recency (days)", fontsize=20)
ax[0].tick_params(axis='x', labelsize=15)

sns.barplot(y="frequency", x="customer_number",
            data=rfm_df.sort_values(by="frequency", ascending=False).head(),
            palette=colors,
            ax=ax[1])

ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title("By Frequency", fontsize=20)
ax[1].tick_params(axis='x', labelsize=15)

sns.barplot(y="monetary",
            x="customer_number",
            data=rfm_df.sort_values(by="monetary", ascending=False).head(),
            palette=colors, ax=ax[2])

ax[2].set_ylabel(None)
ax[2].set_xlabel(None)
ax[2].set_title("Monetary", fontsize=20)
ax[2].tick_params(axis='x', labelsize=15)

plt.suptitle("Cutomer RFM Graph", fontsize=25)

st.pyplot(fig)
