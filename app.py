import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from google import genai

# 1. Page Configuration
st.set_page_config(page_title="Enterprise AI Decisions Engine", layout="wide")
st.title(" Enterprise Analytics & AI Insights Engine")

# 2. Production-Grade Credentials Routing
gemini_api_key = None
if "GEMINI_API_KEY" in st.secrets:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
elif os.environ.get("GEMINI_API_KEY"):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

# 3. Sidebar: Data Ingestion Pipeline
st.sidebar.header("📁 Data Ingestion Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload Corporate Sales Dataset (CSV)", type=["csv"])

@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", end="2026-06-30", freq="D")
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Office Supplies"]
    regions = ["North", "South", "East", "West"]
    
    data = {
        "Date": np.random.choice(dates, size=1500),
        "Category": np.random.choice(categories, size=1500),
        "Region": np.random.choice(regions, size=1500),
        "Sales": np.random.uniform(50, 1500, size=1500),
        "Quantity": np.random.randint(1, 10, size=1500)
    }
    df = pd.DataFrame(data)
    df["Profit"] = df["Sales"] * np.random.uniform(0.15, 0.35, size=1500)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

# Main Data Selection
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        df = raw_df
        st.sidebar.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
        df = generate_mock_data()
else:
    df = generate_mock_data()

# 4. Sidebar: "What-If" Scenario Simulator Layer
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Strategic 'What-If' Simulator")
st.sidebar.caption("Simulate market shocks or strategic reallocations to view projected outcomes.")

target_cat = st.sidebar.selectbox("Select Target Growth Category", df["Category"].unique())
growth_pct = st.sidebar.slider(f"Simulate {target_cat} Sales Shift (%)", -50, 100, 0) / 100.0

target_region = st.sidebar.selectbox("Select Regional Optimization Target", df["Region"].unique())
cost_reduction = st.sidebar.slider(f"Simulate {target_region} Profit Margin Boost (%)", -20, 50, 0) / 100.0

# Apply "What-If" transformations mathematically across the working dataset copy
sim_df = df.copy()
sim_df.loc[sim_df["Category"] == target_cat, "Sales"] *= (1 + growth_pct)
sim_df.loc[sim_df["Category"] == target_cat, "Profit"] *= (1 + growth_pct)
sim_df.loc[sim_df["Region"] == target_region, "Profit"] *= (1 + cost_reduction)

# 5. Core Operational Metric Aggregations
total_sales = sim_df["Sales"].sum()
total_profit = sim_df["Profit"].sum()
total_units = sim_df["Quantity"].sum()
avg_margin = (total_profit / total_sales) * 100

# High-Level Projected Metric Tiles Display
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric("Projected Revenue", f"${total_sales:,.2f}")
kpi_col2.metric("Projected Profit", f"${total_profit:,.2f}")
kpi_col3.metric("Projected Volume", f"{total_units:,} units")
kpi_col4.metric("Projected Margin", f"{avg_margin:.2f}%")

st.markdown("---")

# 6. Core Visualizations Grid
vis_col1, vis_col2 = st.columns(2)
cat_df = sim_df.groupby("Category")["Sales"].sum().reset_index()
region_df = sim_df.groupby("Region")["Profit"].sum().reset_index()

with vis_col1:
    st.subheader("Projected Revenue Distribution by Category")
    fig_bar = px.bar(cat_df, x="Category", y="Sales", color="Category", template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

with vis_col2:
    st.subheader("Projected Regional Profit Contribution")
    fig_pie = px.pie(region_df, values="Profit", names="Region", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# 7. FEATURE: Advanced Statistical Anomaly Detection Layer
st.subheader("🚨 Automated Anomaly & Outlier Tracking Engine")
st.caption("Identifies historical operating dates that deviate significantly from standard sales baselines using Z-Score tracking.")

daily_metrics = sim_df.groupby(sim_df["Date"].dt.date)["Sales"].sum().reset_index()
daily_metrics.columns = ["Date", "Daily_Sales"]

# Apply statistical outlier boundary constraints (Z-Score > 2.0 standard deviations)
mean_sales = daily_metrics["Daily_Sales"].mean()
std_sales = daily_metrics["Daily_Sales"].std()
daily_metrics["Z_Score"] = (daily_metrics["Daily_Sales"] - mean_sales) / std_sales
daily_metrics["Is_Anomaly"] = daily_metrics["Z_Score"].abs() > 2.0

anomalies = daily_metrics[daily_metrics["Is_Anomaly"] == True]

# Build custom multi-scatter graph rendering standard trendlines alongside red anomaly points
fig_trend = px.line(daily_metrics, x="Date", y="Daily_Sales", title="Daily Sales Pipeline Analysis", template="plotly_dark")
if not anomalies.empty:
    fig_trend.add_scatter(
        x=anomalies["Date"], 
        y=anomalies["Daily_Sales"], 
        mode="markers", 
        marker=dict(color="red", size=10, symbol="x"), 
        name="Anomaly Spike/Drop"
    )
st.plotly_chart(fig_trend, use_container_width=True)

# Compile brief string summary of anomaly occurrences to feed directly to Gemini's prompt layer
anomaly_summary_str = "No severe variances flagged."
if not anomalies.empty:
    anomaly_summary_str = ", ".join([f"Date: {row['Date']} (Sales: ${row['Daily_Sales']:,.2f})" for _, row in anomalies.iterrows()])
    st.warning(f"⚠️ **System Alert:** Flagged {len(anomalies)} structural operating variances inside this dataset context.")
else:
    st.success("✅ **System Stable:** No historical data anomalies or volatile pipeline deviations detected.")

st.markdown("---")

# 8. Generative AI Analytics Layer
st.subheader("🤖 AI-Generated Prescriptive Executive Summary")
if not gemini_api_key:
    st.error("🔑 API Key Configuration Missing.")
else:
    @st.cache_data(show_spinner=False)
    def get_prescriptive_report(cat_str, reg_str, metrics_hash, anomalies_str, target_cat_sim, target_reg_sim):
        client = genai.Client(api_key=gemini_api_key)
        prompt_instruction = f"""
        You are a principal corporate data analyst and executive advisor.
        Review the following organizational business metrics, simulated shifts, and data exceptions to compile a prescriptive executive assessment.
        
        Dashboard Operational Metrics:
        {metrics_hash}
        
        Sales Distribution across Categories:
        {cat_str}
        
        Profit Distribution across Regions:
        {reg_str}
        
        Statistical Anomalies Detected:
        {anomalies_str}
        
        Active Simulator Context Modifiers:
        - The user simulated a scenario targeting '{target_cat_sim}' performance adjustments.
        - The user simulated a scenario targeting '{target_reg_sim}' financial structural optimization.
        
        Provide your assessment strictly in the following format:
        ### 1. Performance Overview & Simulated Impact
        (Assess the overall core health of the numbers, explicitly taking into account the simulated changes chosen via the sliders.)
        
        ### 2. Operational Anomalies & Risk Analysis
        (Analyze the statistical data anomalies flagged by the system. If exceptions exist, comment on potential root causes.)
        
        ### 3. Prescriptive Next Steps
        (Provide exactly 3 concise operational actions addressing the data exceptions and simulation parameters.)
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt_instruction)
        return response.text

    metrics_context_hash = f"Rev: {total_sales}, Prof: {total_profit}, Margin: {avg_margin}"
    
    with st.spinner("Analyzing operational matrices and processing simulation models..."):
        summary_text = get_prescriptive_report(
            cat_df.to_string(index=False), 
            region_df.to_string(index=False),
            metrics_context_hash,
            anomaly_summary_str,
            f"Growth for {target_cat}: {growth_pct * 100}%",
            f"Margin shift for {target_region}: {cost_reduction * 100}%"
        )
        with st.container(border=True):
            st.markdown(summary_text)

st.markdown("---")

# 9. Interactive Conversational Assistant (Chatbot)
st.subheader("💬 Chat with Your Business Data")
st.caption("Ask specific contextual questions regarding your uploaded records or simulation impacts.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Pipeline synchronized. I have complete visibility over your metrics, statistical anomalies, and simulated variables. How can I help you today?"}
    ]

# Render chat logs
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture live queries
if user_query := st.chat_input("Ask a question about this file or simulated scenarios..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Querying active memory context..."):
            try:
                data_context = f"""
                You are an internal corporate data assistant. You have access to the following current operational facts:
                - Projected Revenue: ${total_sales:,.2f}
                - Projected Profit: ${total_profit:,.2f}
                - Projected Average Profit Margin: {avg_margin:.2f}%
                
                Active Simulation Settings:
                - {target_cat} growth is adjusted by {growth_pct*100}%
                - {target_region} margin is boosted by {cost_reduction*100}%
                
                Statistical Anomalies Present:
                {anomaly_summary_str}
                
                Category Performance Matrix:
                {cat_df.to_string(index=False)}
                
                Regional Profit Matrix:
                {region_df.to_string(index=False)}
                
                Answer the user's question directly using these facts. Keep answers concise, factual, and data-driven.
                """
                client = genai.Client(api_key=gemini_api_key)
                chat_payload = f"{data_context}\n\nUser Question: {user_query}"
                
                response = client.models.generate_content(model='gemini-2.5-flash', contents=chat_payload)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Chat communication error: {e}")