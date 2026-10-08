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

# Dropdown options from the unique commodity names
all_commodities = wpimonthly_df['COMM_NAME'].dropna().unique()

st.sidebar.subheader("Commodity 1 Selection")
selected_comm1 = st.sidebar.selectbox(
    "Select Primary Commodity:", 
    options=all_commodities, 
    index=list(all_commodities).index('All commodities') if 'All commodities' in all_commodities else 0,
    key="comm1"
)
search_query1 = st.sidebar.text_input("Or Search Commodity 1 Name:", "", key="search1")

st.sidebar.markdown("---")

st.sidebar.subheader("Commodity 2 Selection (Comparison)")
selected_comm2 = st.sidebar.selectbox(
    "Select Comparison Commodity:", 
    options=all_commodities, 
    index=0,  # Defaults to the first item in the list
    key="comm2"
)
search_query2 = st.sidebar.text_input("Or Search Commodity 2 Name:", "", key="search2")

# --- Filter Data Logic ---
# Commodity 1 filtering
if search_query1:
    filtered_data1 = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.contains(search_query1, case=False, na=False)]
    label1 = f"Search: '{search_query1}'"
else:
    filtered_data1 = wpimonthly_df[wpimonthly_df['COMM_NAME'] == selected_comm1]
    label1 = selected_comm1

# Commodity 2 filtering
if search_query2:
    filtered_data2 = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.contains(search_query2, case=False, na=False)]
    label2 = f"Search: '{search_query2}'"
else:
    filtered_data2 = wpimonthly_df[wpimonthly_df['COMM_NAME'] == selected_comm2]
    label2 = selected_comm2

# Combine the data chunks for a single visual layout
combined_filtered_data = pd.concat([filtered_data1, filtered_data2]).drop_duplicates()

st.subheader(f"📊 Comparing: {label1} 🆚 {label2}")

# Display raw filtered dataset slice
with st.expander("View Raw Combined Filtered Data"):
    st.dataframe(combined_filtered_data)

# --- Historical Trend Charting ---
index_cols = [col for col in wpimonthly_df.columns if col.startswith('INDX')]

if not combined_filtered_data.empty:
    # Prepare data for plotting
    plot_df = combined_filtered_data.set_index('COMM_NAME')[index_cols].T
    plot_df.index = pd.to_datetime([col[4:] for col in plot_df.index], format='%m%Y')
    plot_df = plot_df.sort_index()

    # Create matplotlib plot comparing both items
    fig, ax = plt.subplots(figsize=(14, 6))
    for column in plot_df.columns:
        ax.plot(plot_df.index, plot_df[column], label=column, linewidth=2.0)
    
    ax.set_title(f"Monthly Index Trends Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Index Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Commodity Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Streamlit wrapper to display matplotlib plots safely
    st.pyplot(fig)
else:
    st.warning("No data found for the selected filters.")

# --- SARIMAX Modeling Section ---
st.markdown("---")
st.subheader("🤖 Time Series Forecasting (SARIMAX Model)")
st.write("Run the time-series model independently for either of your chosen commodities.")

col_btn1, col_btn2 = st.columns(2)

# Forecast Model for Commodity 1
if col_btn1.button(f"Forecast for '{selected_comm1}'"):
    commodity_ts_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.strip() == selected_comm1]
    
    if not commodity_ts_data.empty:
        with st.spinner(f"Fitting SARIMAX model for {selected_comm1}..."):
            ts_data = commodity_ts_data[index_cols].T
            ts_data.columns = ['Index_Value']
            ts_data.index = pd.to_datetime([col[4:] for col in ts_data.index], format='%m%Y')
            ts_data = ts_data.sort_index()
            ts_data.index.freq = 'MS'
            
            model = SARIMAX(
                ts_data['Index_Value'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_results = model.fit(disp=False)
            
            fig_m1, ax_m1 = plt.subplots(figsize=(8, 4))
            ax_m1.plot(ts_data.index, ts_data['Index_Value'], label='Actual Index')
            ax_m1.plot(ts_data.index, sarimax_results.fittedvalues, color='red', linestyle='--', label='SARIMAX Fitted')
            ax_m1.set_title(f'{selected_comm1}: Actual vs Fitted')
            ax_m1.legend()
            ax_m1.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_m1)
            st.text(f"AIC: {sarimax_results.aic:.2f} | BIC: {sarimax_results.bic:.2f}")
    else:
        st.error(f"Could not fit model for {selected_comm1}.")

# Forecast Model for Commodity 2
if col_btn2.button(f"Forecast for '{selected_comm2}'"):
    commodity_ts_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.strip() == selected_comm2]
    
    if not commodity_ts_data.empty:
        with st.spinner(f"Fitting SARIMAX model for {selected_comm2}..."):
            ts_data = commodity_ts_data[index_cols].T
            ts_data.columns = ['Index_Value']
            ts_data.index = pd.to_datetime([col[4:] for col in ts_data.index], format='%m%Y')
            ts_data = ts_data.sort_index()
            ts_data.index.freq = 'MS'
            
            model = SARIMAX(
                ts_data['Index_Value'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_results = model.fit(disp=False)
            
            fig_m2, ax_m2 = plt.subplots(figsize=(8, 4))
            ax_m2.plot(ts_data.index, ts_data['Index_Value'], label='Actual Index')
            ax_m2.plot(ts_data.index, sarimax_results.fittedvalues, color='blue', linestyle='--', label='SARIMAX Fitted')
            ax_m2.set_title(f'{selected_comm2}: Actual vs Fitted')
            ax_m2.legend()
            ax_m2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_m2)
            st.text(f"AIC: {sarimax_results.aic:.2f} | BIC: {sarimax_results.bic:.2f}")
    else:
        st.error(f"Could not fit model for {selected_comm2}.")
