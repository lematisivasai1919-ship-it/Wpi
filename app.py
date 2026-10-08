import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Page Config ---
st.set_page_config(page_title="WPI Analysis Dashboard", layout="wide")
st.title("📈 Wholesale Price Index (WPI) Analysis Dashboard")

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

# Dropdown to select any unique commodity from the file
all_commodities = wpimonthly_df['COMM_NAME'].dropna().unique()
selected_commodity = st.sidebar.selectbox(
    "Select a Commodity to View Trends:", 
    options=all_commodities, 
    index=list(all_commodities).index('All commodities') if 'All commodities' in all_commodities else 0
)

# Text filter to search commodities (similar to your 'Paddy'/'Copper' search logic)
search_query = st.sidebar.text_input("Or Search Commodity Names (e.g., Paddy, Copper, Gold):", "")

# --- Filter Data ---
if search_query:
    filtered_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.contains(search_query, case=False, na=False)]
    st.subheader(f"🔍 Search Results for: '{search_query}'")
else:
    filtered_data = wpimonthly_df[wpimonthly_df['COMM_NAME'] == selected_commodity]
    st.subheader(f"📊 Visualizing: {selected_commodity}")

# Display raw filtered dataset slice
with st.expander("View Raw Filtered Data"):
    st.dataframe(filtered_data)

# --- Historical Trend Charting ---
index_cols = [col for col in wpimonthly_df.columns if col.startswith('INDX')]

if not filtered_data.empty:
    # Prepare data for plotting
    plot_df = filtered_data.set_index('COMM_NAME')[index_cols].T
    plot_df.index = pd.to_datetime([col[4:] for col in plot_df.index], format='%m%Y')
    plot_df = plot_df.sort_index()

    # Create matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for column in plot_df.columns:
        ax.plot(plot_df.index, plot_df[column], label=column, linewidth=1.5)
    
    ax.set_title(f"Monthly Index Trends", fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index Value')
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

# Let user toggle forecasting for the specific selected commodity
if st.button(f"Run SARIMAX Forecast for '{selected_commodity}'"):
    commodity_ts_data = wpimonthly_df[wpimonthly_df['COMM_NAME'].str.strip() == selected_commodity]
    
    if not commodity_ts_data.empty:
        with st.spinner("Fitting SARIMAX model..."):
            # Structure time series
            ts_data = commodity_ts_data[index_cols].T
            ts_data.columns = ['Index_Value']
            ts_data.index = pd.to_datetime([col[4:] for col in ts_data.index], format='%m%Y')
            ts_data = ts_data.sort_index()
            ts_data.index.freq = 'MS'
            
            # Run model
            model = SARIMAX(
                ts_data['Index_Value'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_results = model.fit(disp=False)
            
            # Layout columns for side-by-side display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot actual vs fitted values
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(ts_data.index, ts_data['Index_Value'], label='Actual Index')
                ax2.plot(ts_data.index, sarimax_results.fittedvalues, color='red', linestyle='--', label='SARIMAX Fitted')
                ax2.set_title(f'SARIMAX Model: Actual vs Fitted Values')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Index Value')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig2)
                
            with col2:
                st.markdown("**Model Fit Summary statistics:**")
                st.text(f"AIC: {sarimax_results.aic:.2f}")
                st.text(f"BIC: {sarimax_results.bic:.2f}")
                st.text(f"HQIC: {sarimax_results.hqic:.2f}")
                with st.expander("View Full SARIMAX Text Summary"):
                    st.text(str(sarimax_results.summary()))
    else:
        st.error("Could not run model. Please select a valid single commodity name.")
