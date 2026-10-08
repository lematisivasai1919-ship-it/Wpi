import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Page Config ---
st.set_page_config(page_title="WPI Analysis Dashboard", layout="wide")
st.title("📈 Wholesale Price Index (WPI) Comparison & Analysis Dashboard")

# --- Load Data Cached to avoid reloading on every click ---
@st.cache_data
def load_data():
    # Load legacy Excel binary format (.xls)
    df = pd.read_excel('monthly_index_202606.csv')
    return df

try:
    wpimonthly_df = load_data()
except Exception as e:
    st.error(f"Error loading data. Make sure 'monthly_index_202606.csv' is in the same directory. Details: {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Interactive Options")

# Fetch unique commodities and insert an empty placeholder at the beginning
raw_commodities = wpimonthly_df['COMM_NAME'].dropna().unique()
dropdown_options = ["-- Empty --"] + list(raw_commodities)

st.sidebar.subheader("Commodity 1 Selection")
selected_comm1 = st.sidebar.selectbox(
    "Select Primary Commodity:", 
    options=dropdown_options, 
    index=dropdown_options.index('All commodities') if 'All commodities' in dropdown_options else 0,
    key="comm1"
)
search_query1 = st.sidebar.text_input("Or Search Commodity 1 Name:", "", key="search1")

st.sidebar.markdown("---")

st.sidebar.subheader("Commodity 2 Selection (Comparison)")
selected_comm2 = st.sidebar.selectbox(
    "Select Comparison Commodity:", 
    options=dropdown_options, 
    index=0,  # Defaults to "-- Empty --"
    key="comm2"
)
search_query2 = st.sidebar.text_input("Or Search Commodity 2 Name:", "", key="search2")

# --- Forecast Steps Option ---
st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting Settings")
forecast_steps = st.sidebar.slider("Months to Forecast into Future:", min_value=1, max_value=24, value=12)

# --- Filter Data Logic ---
data_slices = []
labels = []

# Process Commodity 1
if search_query1:
    filtered_data1 = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.contains(search_query1, case=False, na=False)]
    if not filtered_data1.empty:
        data_slices.append(filtered_data1)
        labels.append(f"Search: '{search_query1}'")
elif selected_comm1 != "-- Empty --":
    filtered_data1 = wpimonthly_df[wpimonthly_df['COMM_NAME'] == selected_comm1]
    if not filtered_data1.empty:
        data_slices.append(filtered_data1)
        labels.append(selected_comm1)

# Process Commodity 2
if search_query2:
    filtered_data2 = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.contains(search_query2, case=False, na=False)]
    if not filtered_data2.empty:
        data_slices.append(filtered_data2)
        labels.append(f"Search: '{search_query2}'")
elif selected_comm2 != "-- Empty --":
    filtered_data2 = wpimonthly_df[wpimonthly_df['COMM_NAME'] == selected_comm2]
    if not filtered_data2.empty:
        data_slices.append(filtered_data2)
        labels.append(selected_comm2)

# --- Combine Data ---
if data_slices:
    combined_filtered_data = pd.concat(data_slices).drop_duplicates()
    display_title = " 🆚 ".join(labels) if len(labels) > 1 else labels
    st.subheader(f"📊 Visualizing: {display_title}")
    
    # Display raw filtered dataset slice
    with st.expander("View Raw Combined Filtered Data"):
        st.dataframe(combined_filtered_data)
else:
    combined_filtered_data = pd.DataFrame()
    st.subheader("📊 Visualizing: No Commodity Selected")

# --- Historical Trend Charting ---
index_cols = [col for col in wpimonthly_df.columns if col.startswith('INDX')]

if not combined_filtered_data.empty:
    # Prepare data for plotting
    plot_df = combined_filtered_data.set_index('COMM_NAME')[index_cols].T
    plot_df.index = pd.to_datetime([col[4:] for col in plot_df.index], format='%m%Y')
    plot_df = plot_df.sort_index()

    # Create matplotlib plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for column in plot_df.columns:
        ax.plot(plot_df.index, plot_df[column], label=column, linewidth=2.0)
    
    ax.set_title(f"Monthly Index Trends", fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Index Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Commodity Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)
else:
    st.warning("Please select at least one commodity from the sidebar options to generate the trend chart.")

# --- SARIMAX Modeling Section ---
st.markdown("---")
st.subheader("🤖 Time Series Forecasting & Future Prediction")

col_btn1, col_btn2 = st.columns(2)

# Forecast Model for Commodity 1
if selected_comm1 != "-- Empty --":
    if col_btn1.button(f"Forecast for '{selected_comm1}'"):
        commodity_ts_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.strip() == selected_comm1]
        
        if not commodity_ts_data.empty:
            with st.spinner(f"Fitting SARIMAX model and predicting future for {selected_comm1}..."):
                ts_data = commodity_ts_data[index_cols].T
                ts_data.columns = ['Index_Value']
                ts_data.index = pd.to_datetime([col[4:] for col in ts_data.index], format='%m%Y')
                ts_data = ts_data.sort_index()
                ts_data.index.freq = 'MS'
                
                # Fit Model
                model = SARIMAX(
                    ts_data['Index_Value'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                sarimax_results = model.fit(disp=False)
                
                # Generate Future Out-Of-Sample Predictions
                forecast_res = sarimax_results.get_forecast(steps=forecast_steps)
                forecast_index = forecast_res.predicted_mean.index
                forecast_values = forecast_res.predicted_mean.values
                
                # Create DataFrame for raw forecasting numbers
                forecast_df = pd.DataFrame({
                    'Predicted Date': forecast_index.strftime('%Y-%m-%d'),
                    'Forecasted Index Value': forecast_values
                }).set_index('Predicted Date')
                
                # Plotting actual vs fitted vs forecast
                fig_m1, ax_m1 = plt.subplots(figsize=(10, 5))
                ax_m1.plot(ts_data.index, ts_data['Index_Value'], label='Actual Historical Index', linewidth=2)
                ax_m1.plot(ts_data.index, sarimax_results.fittedvalues, color='orange', linestyle='--', label='SARIMAX Fitted Model')
                ax_m1.plot(forecast_index, forecast_values, color='red', marker='o', linestyle='-', label=f'Future {forecast_steps}-Month Forecast')
                ax_m1.set_title(f'{selected_comm1}: Historical Data & Future Forecast Trends', fontsize=12, fontweight='bold')
                ax_m1.legend()
                ax_m1.grid(True, linestyle='--', alpha=0.5)
                col_btn1.pyplot(fig_m1)
                
                col_btn1.markdown(f"**Model Stats:** AIC: {sarimax_results.aic:.2f} | BIC: {sarimax_results.bic:.2f}")
                
                # Expandable Raw Forecast Table
                with col_btn1.expander(f"📋 View Raw SARIMAX Forecast Predictions Data ({selected_comm1})"):
                    st.dataframe(forecast_df)
        else:
            col_btn1.error(f"Could not fit model for {selected_comm1}.")
else:
    col_btn1.info("Select a commodity in Primary Selection to run a forecast.")

# Forecast Model for Commodity 2
if selected_comm2 != "-- Empty --":
    if col_btn2.button(f"Forecast for '{selected_comm2}'"):
        commodity_ts_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.strip() == selected_comm2]
        
        if not commodity_ts_data.empty:
            with st.spinner(f"Fitting SARIMAX model and predicting future for {selected_comm2}..."):
                ts_data = commodity_ts_data[index_cols].T
                ts_data.columns = ['Index_Value']
                ts_data.index = pd.to_datetime([col[4:] for col in ts_data.index], format='%m%Y')
                ts_data = ts_data.sort_index()
                ts_data.index.freq = 'MS'
                
                # Fit Model
                model = SARIMAX(
                    ts_data['Index_Value'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                sarimax_results = model.fit(disp=False)
                
                # Generate Future Out-Of-Sample Predictions
                forecast_res = sarimax_results.get_forecast(steps=forecast_steps)
                forecast_index = forecast_res.predicted_mean.index
                forecast_values = forecast_res.predicted_mean.values
                
                # Create DataFrame for raw forecasting numbers
                forecast_df = pd.DataFrame({
                    'Predicted Date': forecast_index.strftime('%Y-%m-%d'),
                    'Forecasted Index Value': forecast_values
                }).set_index('Predicted Date')
                
                # Plotting actual vs fitted vs forecast
                fig_m2, ax_m2 = plt.subplots(figsize=(10, 5))
                ax_m2.plot(ts_data.index, ts_data['Index_Value'], label='Actual Historical Index', linewidth=2)
                ax_m2.plot(ts_data.index, sarimax_results.fittedvalues, color='orange', linestyle='--', label='SARIMAX Fitted Model')
                ax_m2.plot(forecast_index, forecast_values, color='blue', marker='o', linestyle='-', label=f'Future {forecast_steps}-Month Forecast')
                ax_m2.set_title(f'{selected_comm2}: Historical Data & Future Forecast Trends', fontsize=12, fontweight='bold')
                ax_m2.legend()
