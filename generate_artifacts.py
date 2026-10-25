import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Load data
df = pd.read_csv('data/hour.csv')

# Friendly mappings for EDA visualizations
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
weather_map = {1: 'Clear / Few Clouds', 2: 'Mist / Cloudy', 3: 'Light Rain / Snow', 4: 'Heavy Rain / Snow'}

df_eda = df.copy()
df_eda['season_label'] = df_eda['season'].map(season_map)
df_eda['weather_label'] = df_eda['weathersit'].map(weather_map)

# 1. Rentals by Hour
plt.figure(figsize=(10, 5))
hourly_avg = df_eda.groupby('hr')['cnt'].mean().reset_index()
sns.barplot(data=hourly_avg, x='hr', y='cnt', palette='Blues_d')
plt.title('Average Bike Rentals by Hour of the Day (0 - 23)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Hour of the Day (24-hour format)', fontsize=12)
plt.ylabel('Average Number of Rentals', fontsize=12)
plt.tight_layout()
plt.savefig('plots/eda_rentals_by_hour.png', dpi=300)
plt.close()
print("Saved plots/eda_rentals_by_hour.png")

# 2. Rentals by Season
plt.figure(figsize=(8, 5))
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
sns.boxplot(data=df_eda, x='season_label', y='cnt', order=season_order, palette='Set2')
plt.title('Distribution of Bike Rentals Across Seasons', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Total Bike Rentals (cnt)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/eda_rentals_by_season.png', dpi=300)
plt.close()
print("Saved plots/eda_rentals_by_season.png")

# 3. Rentals by Weather Situation
plt.figure(figsize=(9, 5))
sns.barplot(data=df_eda, x='weather_label', y='cnt', estimator=np.mean, errorbar=None, palette='coolwarm')
plt.title('Average Bike Rentals by Weather Condition', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Weather Situation', fontsize=12)
plt.ylabel('Average Rentals', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plots/eda_rentals_by_weather.png', dpi=300)
plt.close()
print("Saved plots/eda_rentals_by_weather.png")

# 4. Working Day vs Holiday / Weekend Rentals by Hour
plt.figure(figsize=(10, 5))
df_eda['Day_Type'] = df_eda['workingday'].map({1: 'Working Day', 0: 'Weekend / Holiday'})
hourly_daytype = df_eda.groupby(['hr', 'Day_Type'])['cnt'].mean().reset_index()
sns.lineplot(data=hourly_daytype, x='hr', y='cnt', hue='Day_Type', style='Day_Type', markers=True, dashes=False, linewidth=2.5, palette=['#1f77b4', '#ff7f0e'])
plt.title('Hourly Bike Rental Patterns: Working Day vs Weekend / Holiday', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Average Rentals', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/eda_rentals_workingday_vs_holiday.png', dpi=300)
plt.close()
print("Saved plots/eda_rentals_workingday_vs_holiday.png")

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'season', 'weathersit', 'cnt']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/eda_correlation_heatmap.png', dpi=300)
plt.close()
print("Saved plots/eda_correlation_heatmap.png")

# 6. Temperature vs Rentals Scatter Plot
plt.figure(figsize=(9, 5))
# Unnormalized temperature in Celsius for intuitive plot: temp * 41
df_eda['temp_celsius'] = df_eda['temp'] * 41
sns.scatterplot(data=df_eda.sample(n=2000, random_state=42), x='temp_celsius', y='cnt', alpha=0.4, color='#2b5c8f')
sns.regplot(data=df_eda.sample(n=2000, random_state=42), x='temp_celsius', y='cnt', scatter=False, color='#d62728', line_kws={'linewidth': 2})
plt.title('Temperature (°C) vs. Total Bike Rentals (with Trendline)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Bike Rentals (cnt)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/eda_temp_vs_rentals.png', dpi=300)
plt.close()
print("Saved plots/eda_temp_vs_rentals.png")

# Model Performance & Residual plots using saved model
model = joblib.load('models/bike_model.pkl')
from sklearn.model_selection import train_test_split
X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_pred = np.maximum(0, model.predict(X_test))
residuals = y_test - y_pred

# 7. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='#1f77b4', edgecolors='none', s=25)
# Perfect prediction line
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction (y = x)')
plt.title('Actual vs. Predicted Bike Rentals (Random Forest)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Actual Rentals (Test Set)', fontsize=12)
plt.ylabel('Predicted Rentals', fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('plots/model_actual_vs_predicted.png', dpi=300)
plt.close()
print("Saved plots/model_actual_vs_predicted.png")

# 8. Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3, color='#2ca02c', edgecolors='none', s=25)
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
plt.title('Residual Plot (Prediction Errors vs. Predicted Values)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Predicted Rentals', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('plots/model_residual_plot.png', dpi=300)
plt.close()
print("Saved plots/model_residual_plot.png")

# 9. Feature Importance Plot
# Extract feature names from OneHotEncoder + numeric cols
preprocessor = model.named_steps['prep']
rf_regressor = model.named_steps['reg']

ohe = preprocessor.named_transformers_['cat']
cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
num_cols = ['temp', 'atemp', 'hum', 'windspeed']

cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = cat_feature_names + num_cols

importances = rf_regressor.feature_importances_
feat_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top10 = feat_df.head(10).iloc[::-1] # reverse for horizontal barplot

# Format feature names for display
readable_names = {
    'hr_17': 'Hour: 5:00 PM (Commute)',
    'hr_18': 'Hour: 6:00 PM (Commute)',
    'temp': 'Temperature (Normalized)',
    'hr_8': 'Hour: 8:00 AM (Commute)',
    'atemp': 'Feeling Temperature (atemp)',
    'yr_1': 'Year 2012 (Growth Trend)',
    'hum': 'Humidity (Normalized)',
    'workingday_1': 'Working Day (Yes)',
    'hr_19': 'Hour: 7:00 PM (Evening)',
    'weathersit_3': 'Weather: Light Rain/Snow',
    'weathersit_1': 'Weather: Clear Skies',
    'season_1': 'Season: Spring'
}
top10['Label'] = top10['Feature'].apply(lambda x: readable_names.get(x, x))

plt.figure(figsize=(10, 6))
bars = plt.barh(top10['Label'], top10['Importance'], color='#3470a3', edgecolor='black', linewidth=0.5)
plt.title('Top 10 Most Important Features (Random Forest Regressor)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Relative Feature Importance (Gini Impurity Reduction)', fontsize=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w*100:.1f}%', va='center', fontsize=10, fontweight='bold')
plt.xlim(0, max(top10['Importance']) * 1.15)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300)
plt.close()
print("Saved plots/feature_importance.png")
print("All artifact figures successfully generated!")
