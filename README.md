# 🚲 Bike Rental Demand Prediction Using Machine Learning
> **Third Year (TY) Computer Engineering Mini Project**  
> **Course:** Machine Learning / Data Science Laboratory  
> **Target:** Hourly Demand Regression (`cnt`)  
> **Champion Algorithm:** Optimized Random Forest Regressor (**R² Score: 0.9284 / 92.8% Accuracy**)

---

## 📄 Project Abstract
Urban bike-sharing systems offer an eco-friendly and flexible public transportation solution across modern smart cities. However, extreme diurnal fluctuations and weather sensitivities often lead to station stockouts during peak commuting hours and idle surplus during off-peak times. This project implements an end-to-end supervised machine learning regression pipeline to forecast hourly bike rental demand using meteorological and temporal variables. Five distinct algorithms—Linear Regression, K-Nearest Neighbors, Decision Trees, Gradient Boosting, and Random Forest—were trained and evaluated using Scikit-Learn Pipelines. After 5-fold cross-validation and hyperparameter optimization, the tuned Random Forest Regressor achieved a state-of-the-art $R^2$ score of $0.9284$ and a Mean Absolute Error of $29.66$ bikes, deployed via an interactive Streamlit web dashboard for real-time fleet operations.

---

## 🎯 Problem Statement & Objectives
Bike-sharing fleet operators face substantial operational losses when bikes are not positioned where demand occurs:
1. **Under-supply:** Commuters arrive at empty docking stations during morning rush hours, causing commuter frustration and lost revenue.
2. **Over-supply:** Evening return stations become full, preventing riders from locking bikes and terminating rentals.
3. **Severe Weather Drops:** Sudden rain or storms drastically decrease rental counts, requiring immediate fleet protection and maintenance scheduling.

### Objectives:
- Perform exhaustive Exploratory Data Analysis (EDA) on 17,379 hours of historical data.
- Prevent **Data Leakage** by strictly eliminating target-derived variables (`casual` and `registered`).
- Build leak-proof Scikit-Learn pipelines combining categorical `OneHotEncoder` and numerical `StandardScaler`.
- Benchmark 5 regression algorithms against standardized metrics ($MAE$, $RMSE$, and $R^2$).
- Perform 5-fold cross-validation hyperparameter tuning using `GridSearchCV`.
- Identify the most influential environmental and temporal drivers of rental volume.
- Deploy an intuitive web user interface using **Streamlit** for live demand forecasting.

---

## 📂 Project Directory Structure

```text
ml-mini-project/
├── data/
│   ├── hour.csv                     # Hourly dataset (17,379 records, 17 features)
│   ├── day.csv                      # Daily aggregated dataset (731 records)
│   └── Readme.txt                   # Official UCI dataset documentation
├── models/
│   └── bike_model.pkl               # Serialized champion pipeline (Joblib)
├── plots/                           # Generated publication-quality figures
│   ├── eda_rentals_by_hour.png
│   ├── eda_rentals_by_season.png
│   ├── eda_rentals_by_weather.png
│   ├── eda_rentals_workingday_vs_holiday.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_temp_vs_rentals.png
│   ├── model_actual_vs_predicted.png
│   ├── model_residual_plot.png
│   └── feature_importance.png
├── app.py                           # Interactive Streamlit Web Application
├── predict.py                       # Reusable prediction module (predict_demand)
├── train.py                         # Standalone model training and tuning script
├── generate_artifacts.py            # High-res visualization generator
├── bike_rental_prediction.ipynb     # Complete Jupyter Notebook (cell-by-cell)
├── requirements.txt                 # Exact project dependencies
├── .gitignore                       # Git exclusion rules
└── README.md                        # Complete project documentation & viva guide
```

---

## 📊 Dataset Details
- **Source:** [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) / [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
- **Log Source:** Capital Bikeshare System (Washington D.C., USA, 2011–2012)
- **Observations:** 17,379 hourly entries
- **Target Variable:** `cnt` (Total count of rented bikes per hour)

### Features Breakdown:
| Feature Name | Type | Description / Categories |
| :--- | :--- | :--- |
| `season` | Categorical | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `yr` | Categorical | 0: 2011, 1: 2012 |
| `mnth` | Categorical | Month of the year (1 to 12) |
| `hr` | Categorical | Hour of the day (0 to 23) |
| `holiday` | Binary | 1: Public holiday, 0: Otherwise |
| `weekday` | Categorical | Day of week (0: Sun, 1: Mon, ..., 6: Sat) |
| `workingday` | Binary | 1: Working day (neither holiday nor weekend), 0: Otherwise |
| `weathersit` | Categorical | 1: Clear, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Snow |
| `temp` | Continuous | Normalized temperature in Celsius ($t / 41$) |
| `atemp` | Continuous | Normalized feeling temperature in Celsius ($at / 50$) |
| `hum` | Continuous | Normalized relative humidity ($h / 100$) |
| `windspeed` | Continuous | Normalized wind speed ($w / 67$) |
| `casual` | Leakage | Count of casual users (DROPPED) |
| `registered` | Leakage | Count of registered users (DROPPED) |
| `cnt` | **Target** | **Total hourly bike rentals (`cnt = casual + registered`)** |

> ⚠️ **Data Leakage Notice:** `casual` and `registered` sum up directly to `cnt`. Including them would cause 100% artificial accuracy during testing but complete failure in real-world deployment. Hence, both are rigorously dropped before training.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
- **Data Visualization:** `matplotlib`, `seaborn`
- **Model Serialization:** `joblib`
- **Web Dashboard:** `streamlit`
- **Interactive Development:** `Jupyter Notebook`

---

## 📈 Exploratory Data Analysis (EDA) Highlights

| Visualization | Key Finding & Operational Takeaway |
| :--- | :--- |
| **Rentals by Hour** | Sharp bimodal distribution peaking at **8:00 AM** (~360 bikes) and **5:00–6:00 PM** (~460 bikes), indicating that office and school commuting drives peak volume. |
| **Working Day vs. Holiday** | Working days exhibit twin commuting spikes. Holidays and weekends exhibit a smooth, midday leisure curve peaking between **1:00 PM and 4:00 PM**. |
| **Rentals by Season** | **Fall** and **Summer** record maximum utilization, while **Spring** exhibits the lowest volume due to prolonged cold precipitation. |
| **Weather Condition** | Clear skies average ~205 rentals/hr, plummeting to ~110 in light rain/snow and under 75 in severe storm conditions. |
| **Correlation Heatmap** | Temperature ($r = +0.40$) and hour ($r = +0.39$) correlate strongest with demand. Humidity has an inverse drag ($r = -0.32$). |
| **Temperature vs. Rentals** | A strong positive linear trajectory exists up to ~30°C, above which extreme heat begins to temper rental volume. |

---

## 🧪 Model Training & Benchmark Results

All models were evaluated on the exact same unseen 20% test partition (3,476 samples) with identical preprocessing pipelines.

| Rank | Model Architecture | MAE (Bikes) | RMSE (Bikes) | $R^2$ Score (Accuracy) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest Regressor (Tuned)** | **29.66** | **47.63** | **0.9284** | **Champion Model** |
| 🥈 | Random Forest Regressor (Default) | 29.79 | 47.83 | 0.9277 | Near-optimal |
| 🥉 | Gradient Boosting Regressor | 57.39 | 79.74 | 0.7992 | Strong non-linear |
| 4 | Decision Tree Regressor (Depth=15) | 52.95 | 80.71 | 0.7943 | High variance |
| 5 | Linear Regression (Baseline) | 69.22 | 97.75 | 0.6982 | Underfitting |
| 6 | K-Nearest Neighbors ($k=5$) | 74.77 | 108.50 | 0.6283 | Curse of dimensionality |

### Optimal Hyperparameters (5-Fold GridSearchCV):
- `reg__n_estimators`: `150`
- `reg__max_depth`: `None` (full expansion controlled by sample split)
- `reg__min_samples_split`: `2`
- `Cross-Validation Score (CV R²)`: `0.928`

---

## 🌟 Top 10 Feature Importances
From the tuned Random Forest model:
1. **Hour: 5:00 PM (18.4%)**: Evening office departure rush.
2. **Hour: 6:00 PM (12.2%)**: Secondary evening commute wave.
3. **Temperature (11.5%)**: Primary environmental comfort factor.
4. **Hour: 8:00 AM (9.8%)**: Morning rush hour peak.
5. **Feeling Temperature (`atemp`) (8.1%)**: Perceived heat/windchill.
6. **Year 2012 (`yr_1`) (6.9%)**: System adoption and bike fleet expansion.
7. **Humidity (5.4%)**: High humidity deters outdoor physical cycling.
8. **Working Day (3.8%)**: Separates dual-peak commuters from recreational riders.
9. **Hour: 7:00 PM (3.1%)**: Evening leisure and late commuter transit.
10. **Weather Situation: Clear (2.6%)**: Favorable riding condition.

---

## 💻 How to Run the Project Locally

### 1. Prerequisites
Ensure you have **Python 3.10 to 3.14** installed on your system.

### 2. Clone or Navigate to the Repository
```bash
cd "c:\Users\Rutikesh\ml mini project"
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Train the Model (Optional - Pre-trained model included)
To re-run the 5-fold cross-validation and generate `models/bike_model.pkl`:
```bash
python train.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

### 6. Run the Jupyter Notebook
```bash
jupyter notebook bike_rental_prediction.ipynb
```

---

## 🔮 Reusable Python API

You can import and use the prediction engine in any Python script:

```python
from predict import predict_demand

# Input dictionary with human-friendly values (auto-normalizes internally)
sample_query = {
    'season': 2,        # 1: Spring, 2: Summer, 3: Fall, 4: Winter
    'yr': 1,            # 0: 2011, 1: 2012
    'mnth': 6,          # June (1-12)
    'hr': 17,           # 5:00 PM (0-23)
    'holiday': 0,       # 0: No, 1: Yes
    'weekday': 1,       # 1: Monday (0-6)
    'workingday': 1,    # 1: Yes, 0: No
    'weathersit': 1,    # 1: Clear, 2: Mist, 3: Light Rain, 4: Heavy Rain
    'temp': 28.0,       # Temperature in °C (or normalized 0.0-1.0)
    'atemp': 29.0,      # Feeling temperature in °C
    'hum': 45.0,        # Humidity percentage (0-100%)
    'windspeed': 12.0   # Wind speed in km/h
}

prediction = predict_demand(sample_query)
print(f"Predicted Demand: {prediction['predicted_count']} bikes/hr")
print(f"Demand Category: {prediction['demand_level']}")
print(f"Expected Range: {prediction['confidence_interval_low']} to {prediction['confidence_interval_high']} bikes")
```

---

## 🎓 Viva Voce Preparation (10 Common Questions & Answers)

### Q1: Why did Random Forest outperform Linear Regression and single Decision Trees?
**Answer:** Linear Regression is constrained by the assumption of linearity and fails to capture complex non-linear interactions (e.g., commute hour spikes vs. weekend leisure curves). Single Decision Trees capture non-linear patterns but suffer from high variance and overfit to noise. Random Forest uses **Bagging (Bootstrap Aggregation)** with feature subsampling across 150 diverse trees. Averaging their predictions cancels out individual tree errors, drastically reducing variance while maintaining low bias ($R^2 = 0.928$).

### Q2: What is Data Leakage, and why were `casual` and `registered` dropped?
**Answer:** Data leakage occurs when variables that would not be available at inference time are included in training, falsely inflating accuracy. In this dataset, total rentals (`cnt`) is mathematically the exact sum of `casual + registered`. If kept, the model trivially learns the addition operator rather than environmental demand patterns, leading to 100% false accuracy during training and total failure in production where neither number is known beforehand.

### Q3: What is the mathematical and practical difference between MAE and RMSE?
**Answer:**
- **MAE (Mean Absolute Error):** $\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$. Measures average absolute magnitude of error on a linear scale.
- **RMSE (Root Mean Squared Error):** $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$. Squares the errors before averaging and taking the root.
Because RMSE squares individual errors, it penalizes large errors much more severely. For fleet management, large under-predictions leave hundreds of commuters stranded, making RMSE a critical operational metric.

### Q4: What does an $R^2$ score of 0.9284 represent?
**Answer:** The Coefficient of Determination ($R^2$) represents the proportion of variance in the dependent variable explained by the independent features:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
An $R^2$ of $0.9284$ means our model explains **92.84% of the total variance** in hourly bike rentals. Only 7.16% remains unexplained, which accounts for unmeasured factors like local traffic accidents, metro delays, or sporadic events.

### Q5: Why is One-Hot Encoding necessary for categorical columns like `season` or `weekday`?
**Answer:** If encoded as integers ($1, 2, 3, 4$), regression and distance algorithms assume an inherent mathematical ordering ($4 > 1$), falsely implying that Winter is 4 times "greater" than Spring. One-Hot Encoding converts each category into independent binary flags ($0$ or $1$), ensuring equal nominal distance and preventing artificial numerical bias.

### Q6: Why use Scikit-Learn `Pipeline` and `ColumnTransformer`?
**Answer:** Pipelines bundle preprocessing steps (`OneHotEncoder`, `StandardScaler`) and estimators into a single atomic object. This strictly prevents data leakage during cross-validation (transformers are fitted solely on training folds) and guarantees that new production inputs undergo identical transformations during deployment without error-prone manual transformations.

### Q7: What is Overfitting, and how did you verify your model generalizes well?
**Answer:** Overfitting happens when a model memorizes noise in the training set and fails on unseen data. We verified our model by running 5-fold cross validation during hyperparameter tuning (average CV $R^2 \approx 0.928$) and evaluating on a strictly isolated 20% holdout test set ($R^2 = 0.9284$). The close parity between cross-validation and test scores proves strong generalization.

### Q8: Why did KNN Regressor perform the worst ($R^2 = 0.628$)?
**Answer:** KNN calculates Euclidean distances between data points. After one-hot encoding categorical variables (hours, months, weekdays, seasons), the feature space expands to 57 dimensions. In high-dimensional spaces (**curse of dimensionality**), data points become sparse and equidistant, diluting the concept of nearest neighbors.

### Q9: What is the role of K-Fold Cross-Validation?
**Answer:** In 5-fold cross-validation, the training set is split into 5 equal subsets. The model trains on 4 folds and evaluates on the 5th, iterating 5 times so each fold acts as the validation set once. This eliminates bias from a single lucky or unlucky split and ensures robust parameter selection during `GridSearchCV`.

### Q10: What is the real-world engineering impact of this system in Smart Cities?
**Answer:** Predicting demand hours in advance empowers municipal transit authorities to:
1. Proactively dispatch mobile rebalancing trucks to transport bikes from surplus stations to high-demand commuter hubs.
2. Schedule docking station maintenance and battery swaps during low-demand night hours.
3. Implement dynamic surge discounts to incentivize users to ride bikes toward depleted stations.

---

## 🚀 GitHub Upload Instructions

Run the following commands in PowerShell from the project root:

```powershell
# 1. Initialize local Git repository
git init

# 2. Add all project files
git add .

# 3. Commit changes with a descriptive message
git commit -m "feat: complete bike rental demand prediction mini project with streamlit app"

# 4. Set default branch to main
git branch -M main

# 5. Add remote GitHub repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/bike-rental-demand-prediction.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 🔮 Future Scope
- **Real-Time Weather API Integration:** Connect OpenWeatherMap API to fetch live weather conditions automatically.
- **Deep Learning Comparison:** Benchmark against LSTM / Temporal Fusion Transformers (TFT) for sequential time-series forecasting.
- **Station-Level Spatial Prediction:** Extend from citywide aggregation to individual geo-spatial station docking docks using Graph Neural Networks (GNN).
- **Dynamic Pricing Recommendation:** Suggest micro-incentives to users for rerouting bikes to empty hubs.

---

## 👨‍💻 Author & Acknowledgements
- **Author:** Rutikesh
- **Degree:** Third Year (TY) Bachelor of Technology / Engineering in Computer Engineering
- **Dataset Acknowledgement:** Hadi Fanaee-T & João Gama, *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence, Springer (2013).
