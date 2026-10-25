"""
Bike Rental Demand Prediction - Streamlit Web Application
TY Computer Engineering Mini Project

Run using:
    streamlit run app.py
"""

import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Import our reusable prediction function
from predict import predict_demand, load_model

# Page Configuration
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and typography
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
        margin-top: 5px;
    }
    .badge-peak {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-mod {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #6B7280;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🚲 Bike Rental Demand Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">TY Computer Engineering Mini Project • Supervised Machine Learning (Random Forest Regressor, R² ≈ 0.93)</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_pred, tab_eda, tab_model, tab_viva = st.tabs([
    "🔮 Live Demand Predictor", 
    "📊 Exploratory Data Analysis (EDA)", 
    "📈 Model Benchmarks & Metrics", 
    "🎓 Viva & Theory Questions"
])

# ==============================================================================
# TAB 1: LIVE DEMAND PREDICTOR
# ==============================================================================
with tab_pred:
    st.subheader("Enter Environmental and Calendar Parameters")
    st.write("Adjust the inputs below to predict expected bike rentals for any chosen hour.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📅 Calendar & Temporal Settings")
        
        # Date selection
        selected_date = st.date_input(
            "Select Date",
            value=datetime.date(2026, 6, 15),
            min_value=datetime.date(2010, 1, 1),
            max_value=datetime.date(2030, 12, 31),
            help="Year, month, and day-of-week will be automatically derived."
        )
        
        # Hour slider
        selected_hour = st.slider(
            "Hour of Day (0:00 to 23:00)",
            min_value=0,
            max_value=23,
            value=8,
            step=1,
            format="%d:00",
            help="Peak hours typically occur around 8:00 AM and 5:00-6:00 PM on workdays."
        )

        # Season
        season_options = {
            "Spring (Mar - May)": 1,
            "Summer (Jun - Aug)": 2,
            "Fall (Sep - Nov)": 3,
            "Winter (Dec - Feb)": 4
        }
        # Smart default season based on selected month
        def_month = selected_date.month
        if def_month in [3, 4, 5]:
            def_season_idx = 0
        elif def_month in [6, 7, 8]:
            def_season_idx = 1
        elif def_month in [9, 10, 11]:
            def_season_idx = 2
        else:
            def_season_idx = 3

        selected_season_name = st.selectbox(
            "Season",
            options=list(season_options.keys()),
            index=def_season_idx
        )
        selected_season = season_options[selected_season_name]

        # Day flags
        weekday_idx = selected_date.weekday() # 0 = Monday, 6 = Sunday
        # In dataset: 0: Sunday, 1: Monday, ..., 6: Saturday
        dataset_weekday = (weekday_idx + 1) % 7 
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            is_holiday = st.checkbox("Public Holiday", value=False)
        with col_f2:
            # Default working day is True if Mon-Fri and not holiday
            default_working = (weekday_idx < 5) and (not is_holiday)
            is_workingday = st.checkbox("Working Day", value=default_working)

    with col_right:
        st.markdown("#### 🌦️ Weather & Atmospheric Conditions")

        weather_options = {
            "Clear / Few Clouds / Partly Cloudy": 1,
            "Mist / Cloudy / Broken Clouds": 2,
            "Light Snow / Light Rain / Thunderstorm": 3,
            "Heavy Rain / Ice Pellets / Thunderstorm / Snow": 4
        }
        selected_weather_name = st.selectbox(
            "Weather Condition",
            options=list(weather_options.keys()),
            index=0
        )
        selected_weather = weather_options[selected_weather_name]

        # Temperature
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_c = st.slider("Temperature (°C)", min_value=-5.0, max_value=40.0, value=24.0, step=0.5)
        with col_t2:
            atemp_c = st.slider("Feeling Temperature (°C)", min_value=-5.0, max_value=45.0, value=25.0, step=0.5)

        # Humidity & Windspeed
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=52.0, step=1.0)
        with col_w2:
            windspeed_kmh = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=1.0)

        # Input validation check
        if selected_weather == 4:
            st.warning("⚠️ Severe weather alert: Heavy Rain/Snow will significantly suppress rental demand.")

    # Year mapping: 0 for 2011, 1 for 2012 or later
    yr_encoded = 1 if selected_date.year >= 2012 else 0

    st.markdown("---")

    # Predict Button
    predict_clicked = st.button("🚀 Predict Bike Rental Demand", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            # Build input dictionary
            features = {
                'season': selected_season,
                'yr': yr_encoded,
                'mnth': selected_date.month,
                'hr': selected_hour,
                'holiday': 1 if is_holiday else 0,
                'weekday': dataset_weekday,
                'workingday': 1 if is_workingday else 0,
                'weathersit': selected_weather,
                'temp': temp_c,
                'atemp': atemp_c,
                'hum': humidity,
                'windspeed': windspeed_kmh
            }

            # Run prediction
            with st.spinner("Executing model pipeline..."):
                output = predict_demand(features)

            count = output['predicted_count']
            level = output['demand_level']
            low_ci = output['confidence_interval_low']
            high_ci = output['confidence_interval_high']

            # Display Results Cards
            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

            with res_col1:
                badge_class = {
                    'Peak': 'badge-peak',
                    'High': 'badge-high',
                    'Moderate': 'badge-mod',
                    'Low': 'badge-low'
                }.get(level, 'badge-mod')

                st.markdown(f"""
                <div class="metric-card">
                    <span class="{badge_class}">DEMAND LEVEL: {level.upper()}</span>
                    <div class="metric-value">{count}</div>
                    <div class="metric-label">Predicted Bike Rentals / Hour</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.metric(
                    label="Expected Uncertainty Range",
                    value=f"{low_ci} - {high_ci} bikes",
                    delta=f"±47 bikes (RMSE)"
                )
                st.info(f"**Day Type**: {'Working Day' if is_workingday else 'Holiday/Weekend'}\n\n**Hour**: {selected_hour:02d}:00")

            with res_col3:
                # Operational recommendations
                st.markdown("##### 🚲 Fleet Management Insight")
                if level == 'Peak':
                    st.error("🚨 **High Fleet Rebalancing Required**: Pre-stage bikes at transit hubs and metro stations.")
                elif level == 'High':
                    st.warning("⚡ **High Demand**: Ensure active docking stations have at least 80% capacity.")
                elif level == 'Moderate':
                    st.success("✅ **Balanced Fleet**: Standard operational schedule sufficient.")
                else:
                    st.info("🌙 **Low Demand Window**: Ideal period for maintenance and battery recharging.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.subheader("Key Insights from Exploratory Data Analysis")
    st.write("Visualizations derived from 17,379 hourly records of the Capital Bikeshare dataset.")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("##### 1. Rentals by Hour of the Day")
        if os.path.exists("plots/eda_rentals_by_hour.png"):
            st.image("plots/eda_rentals_by_hour.png", use_container_width=True)
            st.caption("💡 **Insight:** Demand exhibits sharp bimodal commute peaks at 8:00 AM (rush hour) and 5:00-6:00 PM (office departure).")
        
        st.markdown("##### 2. Rentals by Weather Condition")
        if os.path.exists("plots/eda_rentals_by_weather.png"):
            st.image("plots/eda_rentals_by_weather.png", use_container_width=True)
            st.caption("💡 **Insight:** Clear skies average ~205 rentals/hr, dropping dramatically to ~110 in light rain/snow and under 75 in severe storms.")

        st.markdown("##### 3. Correlation Heatmap")
        if os.path.exists("plots/eda_correlation_heatmap.png"):
            st.image("plots/eda_correlation_heatmap.png", use_container_width=True)
            st.caption("💡 **Insight:** Temperature and hour have the highest positive correlation with demand, while humidity is negatively correlated (-0.32).")

    with eda_col2:
        st.markdown("##### 4. Working Days vs. Weekends / Holidays")
        if os.path.exists("plots/eda_rentals_workingday_vs_holiday.png"):
            st.image("plots/eda_rentals_workingday_vs_holiday.png", use_container_width=True)
            st.caption("💡 **Insight:** Working days exhibit twin commuter spikes; weekends exhibit a smooth, midday leisure peak between 1:00 PM and 4:00 PM.")

        st.markdown("##### 5. Seasonal Rental Distribution")
        if os.path.exists("plots/eda_rentals_by_season.png"):
            st.image("plots/eda_rentals_by_season.png", use_container_width=True)
            st.caption("💡 **Insight:** Fall and Summer observe maximum bike rentals, while Spring exhibits the lowest volume due to cold precipitation.")

        st.markdown("##### 6. Temperature vs. Rentals (Scatter with Trend)")
        if os.path.exists("plots/eda_temp_vs_rentals.png"):
            st.image("plots/eda_temp_vs_rentals.png", use_container_width=True)
            st.caption("💡 **Insight:** A steady positive linear relationship is observed between ambient temperature and rental volume up to ~32°C.")

# ==============================================================================
# TAB 3: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab_model:
    st.subheader("Model Comparison & Performance Benchmarks")
    st.write("All models were trained on 80% of data (13,903 samples) and evaluated on an unseen 20% test set (3,476 samples).")

    # Benchmark Table
    metrics_data = {
        "Algorithm": [
            "Random Forest Regressor (Tuned)",
            "Random Forest Regressor (Default)",
            "Gradient Boosting Regressor",
            "Decision Tree Regressor",
            "Linear Regression (Baseline)",
            "K-Nearest Neighbors (KNN)"
        ],
        "MAE (Bikes)": [29.66, 29.79, 57.39, 52.95, 69.22, 74.77],
        "RMSE (Bikes)": [47.63, 47.83, 79.74, 80.71, 97.75, 108.50],
        "R² Score": [0.9284, 0.9277, 0.7992, 0.7943, 0.6982, 0.6283],
        "Status": ["🏆 Best Model", "Near Optimal", "Strong", "Moderate", "Baseline", "Weakest"]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("##### Actual vs. Predicted Rentals (Random Forest)")
        if os.path.exists("plots/model_actual_vs_predicted.png"):
            st.image("plots/model_actual_vs_predicted.png", use_container_width=True)
            st.caption("Points cluster tightly along the 45-degree diagonal red line (y = x), indicating high accuracy (R² = 0.928).")

        st.markdown("##### Top 10 Feature Importances")
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", use_container_width=True)
            st.caption("Commute hours (5 PM, 6 PM, 8 AM), ambient temperature, and year-over-year growth are the top drivers of rental demand.")

    with m_col2:
        st.markdown("##### Residual Plot (Error vs. Predicted)")
        if os.path.exists("plots/model_residual_plot.png"):
            st.image("plots/model_residual_plot.png", use_container_width=True)
            st.caption("Residuals are symmetrically distributed around zero with slight heteroscedasticity at very high demand volumes (> 700).")

# ==============================================================================
# TAB 4: VIVA PREPARATION & THEORY
# ==============================================================================
with tab_viva:
    st.subheader("🎓 Mini Project Viva Q&A Guide")
    st.write("Comprehensive answers to the 10 most common external examiner questions.")

    viva_qa = [
        ("1. Why did Random Forest outperform Linear Regression and Decision Tree?",
         "Linear Regression assumes a strictly linear relationship between features and bike rentals, failing to capture complex non-linear patterns (e.g., commute hour spikes). Single Decision Trees capture non-linearities but suffer from high variance (overfitting). Random Forest is an ensemble of 150 bagged decision trees trained on bootstrap samples with feature subsampling, which drastically reduces variance and yields an R² score of ~0.93."),
         
        ("2. What is Data Leakage and why were 'casual' and 'registered' dropped?",
         "Data leakage occurs when information from the target variable is inadvertently present in the features. In this dataset, total rentals (cnt) is mathematically the exact sum of casual rentals + registered rentals (`cnt = casual + registered`). If we kept those columns, the model would simply learn to add two numbers, yielding an artificial 100% accuracy that would fail in production where neither number is known beforehand."),

        ("3. What is the difference between MAE and RMSE? Which should be prioritized?",
         "Mean Absolute Error (MAE) computes the average magnitude of errors without considering direction: MAE = (1/n) * Σ |y - ŷ|. Root Mean Squared Error (RMSE) squares errors before averaging: RMSE = sqrt((1/n) * Σ (y - ŷ)²). RMSE penalizes large outlier errors much more heavily due to the squaring operation. For bike rental management, RMSE is critical because large underpredictions cause empty stations and stranded commuters."),

        ("4. What does an R² score of 0.928 mean in plain English?",
         "The Coefficient of Determination (R²) represents the proportion of variance in the dependent variable (bike rentals) explained by the independent features in the model. An R² of 0.928 means that 92.8% of the variability in hourly bike rental demand is successfully explained by our weather and calendar features, with only 7.2% due to random unmeasured noise."),

        ("5. How does One-Hot Encoding work and why is it necessary for categorical columns?",
         "Machine learning algorithms interpret integer numbers mathematically (e.g., 4 > 1). If we pass season as 1 (Spring) and 4 (Winter), a model might incorrectly assume Winter is 4x 'greater' than Spring. One-Hot Encoding converts each categorical category into a binary column (0 or 1), ensuring the model treats categories with equal nominal distance without imposing an artificial hierarchy."),

        ("6. Why use Scikit-Learn Pipelines instead of manual transformations?",
         "Pipelines encapsulate the entire workflow—preprocessing (encoding + scaling) and model fitting—into a single atomic object. This prevents data leakage (scalers fit only on training data), ensures identical transformation during testing/deployment, and simplifies deployment via a single serialized file (`joblib.dump`)."),

        ("7. What is Overfitting and how do you verify your model isn't overfitted?",
         "Overfitting occurs when a model learns the training noise rather than the underlying pattern, performing great on training data but poorly on unseen test data. We verify our model by comparing 5-fold cross-validation scores with unseen test set scores (train R² ~ 0.98, test R² ~ 0.93), showing strong generalization without severe degradation."),

        ("8. Why did KNN Regressor perform poorly compared to Tree models?",
         "KNN calculates Euclidean distance across all dimensions. After one-hot encoding categorical variables (seasons, months, hours, weekdays), the feature space expands to ~57 dimensions. In high-dimensional space, the 'curse of dimensionality' makes all data points equidistant, degrading distance-based neighbor searches."),

        ("9. What is K-Fold Cross Validation?",
         "In 5-fold cross validation, the training dataset is partitioned into 5 equal folds. The model is trained on 4 folds and validated on the remaining 1 fold, repeated 5 times so every fold acts as a test set once. The average score provides an unbiased estimate of generalization performance and prevents lucky/unlucky train-test splits."),

        ("10. What is the business value of predicting bike rentals for smart cities?",
         "Bike sharing operators face severe supply-demand imbalances: empty stations during morning rush hours and overflowing docks at destination stations. Accurately predicting demand 1 to 24 hours in advance allows operators to proactively dispatch rebalancing vans, schedule maintenance during low-demand night hours, and dynamically adjust pricing.")
    ]

    for q, a in viva_qa:
        with st.expander(f"📌 {q}", expanded=False):
            st.write(a)
