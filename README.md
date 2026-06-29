# 📊 Enterprise AI Analytics & Strategic Decisions Engine

An interactive, production-ready Business Intelligence (BI) dashboard built with Python and Streamlit. This platform processes transactional sales records, visualizes key performance distribution matrices, flags statistical anomalies via zero-dependency Z-Score tracking, and runs predictive "What-If" simulations. It utilizes the native Google GenAI SDK to securely ground Gemini 2.5 on active data frames for automated executive strategy reports and conversational analytics chatting.

---

## ⚡ Key Features

*   📁 **Dynamic Ingestion Pipeline:** Accepts any standard corporate sales CSV or scales automatically using a baseline structural mock data generator if no file is present.
*   🎛️ **Predictive "What-If" Simulator:** Front-end slider controls dynamically map conditional transformations across underlying Pandas data frames to forecast operational revenue shifts instantly.
*   🚨 **Statistical Anomaly Tracker:** Implements a localized Z-Score mathematical boundary check ($Z > 2.0$) on historical sales to flag and isolate volatile market anomalies.
*   🤖 **Grounded Executive Reporting:** Combines aggregated data metrics with the `gemini-2.5-flash` model via the official Google GenAI SDK to generate structured prescriptive summaries.
*   💬 **Conversational Data Chat:** A multi-turn chat assistant built using Streamlit's state cache management (`st.session_state`) that allows administrators to query active dashboard data fields using natural language.

---

## 🛠️ Tech Stack & Architecture

*   **UI Framework:** Streamlit (Layout components, inputs, and session lifecycle tracking)
*   **Data Pipeline:** Pandas & NumPy (Data engineering, array transformations, aggregation logic)
*   **Data Visualizations:** Plotly Express (Dynamic multi-trace charts and data trends)
*   **AI Engine:** Google GenAI SDK (`gemini-2.5-flash`)
*   **Security:** Decoupled architecture using local TOML configuration files for pipeline API tokens

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/Ashish-eer/ai-executive-insights-dashboard.git](https://github.com/Ashish-eer/ai-executive-insights-dashboard.git)
cd ai-executive-insights-dashboard
