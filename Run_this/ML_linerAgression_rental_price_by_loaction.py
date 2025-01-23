import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the CSV file
file_path = '/Users/ethan/python_learn/Dataset/2024_rental_prices_tokyo.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {str(e)}")
    exit()

# Calculate cost per square meter
try:
    data['cost_per_sqm'] = data['cost'] / data['size']
except KeyError:
    print("Error: 'cost' or 'size' column not found in the CSV file.")
    exit()
except ZeroDivisionError:
    print("Error: Found zero values in 'size' column.")
    # Remove rows with zero size or handle differently as needed
    data = data[data['size'] > 0]

# Display first few rows and data info
print("First few rows of the dataset:")
print(data.head())
print("\nDataset information:")
print(data.info())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Convert categorical station data to numerical using Label Encoding
data['station'] = data['station'].fillna('Unknown')
label_encoder = LabelEncoder()
try:
    station_encoded = label_encoder.fit_transform(data['station'])
except KeyError:
    print("Error: 'station' column not found in the CSV file.")
    exit()
except Exception as e:
    print(f"Error encoding station data: {str(e)}")
    exit()

# Reshape data for sklearn
X = station_encoded.reshape(-1, 1)
y = data['cost_per_sqm']  # Using cost per square meter instead of absolute cost

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X)

# Calculate average cost per square meter by station for visualization
station_avg_cost = data.groupby('station')['cost_per_sqm'].mean().reset_index()
station_avg_cost['station_encoded'] = label_encoder.transform(station_avg_cost['station'])

# Create the visualization
plt.figure(figsize=(12, 8))

# Scatter plot of actual data
plt.scatter(X, y, color='blue', label='Actual Prices (per sq.m)', alpha=0.5)

# Line plot of predicted values
sorted_X = np.array(sorted(X)).reshape(-1, 1)
plt.plot(sorted_X, model.predict(sorted_X), color='red', label='Predicted Trend')

# Customize the plot
plt.xlabel('Station (Encoded)')
plt.ylabel('Cost per Square Meter (¥)')
plt.title('Tokyo Rental Prices per Square Meter by Station')
plt.legend()

# Add station labels
for idx, row in station_avg_cost.iterrows():
    plt.annotate(row['station'],
                (row['station_encoded'], row['cost_per_sqm']),
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# Print model statistics
print("\nModel Performance:")
print(f"Model Score (R²): {model.score(X, y):.3f}")
print("\nAverage Cost per Square Meter by Station:")
print(station_avg_cost[['station', 'cost_per_sqm']].to_string(index=False))

# Save the plot
plt.savefig('tokyo_rent_analysis_per_sqm.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional statistics
print("\nDetailed Statistics by Station:")
print(data.groupby('station')['cost_per_sqm'].agg(['count', 'mean', 'std', 'min', 'max']))



# Create two separate figures for our analyses
plt.style.use('seaborn')

# 1. Polynomial Regression Analysis
print("\n=== Polynomial Regression Analysis ===")
label_encoder = LabelEncoder()
station_encoded = label_encoder.fit_transform(data['station']).reshape(-1, 1)

# Create polynomial features (trying degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(station_encoded)

# Fit polynomial regression
poly_model = LinearRegression()
poly_model.fit(X_poly, data['cost_per_sqm'])
y_pred_poly = poly_model.predict(X_poly)

# Calculate average cost per square meter by station for visualization
station_avg_cost = data.groupby('station')['cost_per_sqm'].agg(['mean', 'count']).reset_index()
station_avg_cost['station_encoded'] = label_encoder.transform(station_avg_cost['station'])

# Create polynomial regression plot
plt.figure(figsize=(15, 8))
plt.scatter(station_encoded, data['cost_per_sqm'], alpha=0.3, label='Actual Prices')

# Sort for smooth curve plotting
sort_idx = np.argsort(station_encoded.ravel())
plt.plot(station_encoded[sort_idx], y_pred_poly[sort_idx], color='red', label='Polynomial Trend')

plt.xlabel('Station (Encoded)')
plt.ylabel('Cost per Square Meter (¥)')
plt.title('Tokyo Rental Prices per Square Meter by Station (Polynomial Regression)')
plt.legend()

# Add station labels for top 5 most expensive and 5 least expensive stations
top_stations = station_avg_cost.nlargest(5, 'mean')
bottom_stations = station_avg_cost.nsmallest(5, 'mean')
stations_to_label = pd.concat([top_stations, bottom_stations])

for _, row in stations_to_label.iterrows():
    plt.annotate(f"{row['station']}\n(¥{row['mean']:,.0f}/m²)",
                (row['station_encoded'], row['mean']),
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.savefig('tokyo_rent_analysis_polynomial.png', dpi=300, bbox_inches='tight')
plt.show()

# Create visualization for all three models
plt.style.use('seaborn')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24))

# Plot 1: Linear Regression
ax1.scatter(X, y, color='blue', alpha=0.3, label='Actual Prices')
sorted_X = np.array(sorted(X)).reshape(-1, 1)
ax1.plot(sorted_X, model.predict(sorted_X), color='red', label='Linear Trend')
ax1.set_xlabel('Station (Encoded)')
ax1.set_ylabel('Cost per Square Meter (¥)')
ax1.set_title('Linear Regression Analysis')
ax1.legend()

# Plot 2: Polynomial Regression
ax2.scatter(X, y, alpha=0.3, label='Actual Prices')
sort_idx = np.argsort(X.ravel())
ax2.plot(X[sort_idx], y_pred_poly[sort_idx], color='red', label='Polynomial Trend')
ax2.set_xlabel('Station (Encoded)')
ax2.set_ylabel('Cost per Square Meter (¥)')
ax2.set_title('Polynomial Regression Analysis')
ax2.legend()
# Plot 3: One-Hot Encoding Effects
station_effects = pd.DataFrame({
    'station': [col.replace('station_', '') for col in station_dummies.columns],
    'price_effect': onehot_model.coef_
})
station_effects_sorted = station_effects.sort_values('price_effect', ascending=False)
top_bottom_stations = pd.concat([
    station_effects_sorted.head(10),
    station_effects_sorted.tail(10)
])

colors = ['green']*10 + ['red']*10
ax3.bar(range(20), top_bottom_stations['price_effect'], color=colors)
ax3.set_xticks(range(20))
ax3.set_xticklabels(top_bottom_stations['station'], rotation=45, ha='right')
ax3.set_xlabel('Station')
ax3.set_ylabel('Price Effect (¥/m²)')
ax3.set_title('Top 10 Most Expensive and Least Expensive Stations (One-Hot Encoding)')

plt.tight_layout()
plt.savefig('tokyo_rent_analysis_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# Print model performance metrics
print("\nModel Performance Metrics:")
print(f"Linear Regression R²: {model.score(X, y):.3f}")
print(f"Polynomial Regression R²: {poly_model.score(X_poly, y):.3f}")
print(f"One-Hot Encoding R²: {onehot_model.score(station_dummies, data['cost_per_sqm']):.3f}")
print("\n=== One-Hot Encoding Analysis ===")
# Create one-hot encoding for stations
station_dummies = pd.get_dummies(data['station'], prefix='station')

# Fit linear regression with one-hot encoded features
onehot_model = LinearRegression()
onehot_model.fit(station_dummies, data['cost_per_sqm'])

# Get station coefficients and create a DataFrame for analysis
station_effects = pd.DataFrame({
    'station': [col.replace('station_', '') for col in station_dummies.columns],
    'price_effect': onehot_model.coef_
})

# Sort stations by price effect
station_effects_sorted = station_effects.sort_values('price_effect', ascending=False)

# Create bar plot for top and bottom 10 stations
plt.figure(figsize=(15, 8))
top_bottom_stations = pd.concat([
    station_effects_sorted.head(10),
    station_effects_sorted.tail(10)
])

colors = ['green']*10 + ['red']*10
plt.bar(range(20), top_bottom_stations['price_effect'], color=colors)
plt.xticks(range(20), top_bottom_stations['station'], rotation=45, ha='right')
plt.xlabel('Station')
plt.ylabel('Price Effect (¥/m²)')
plt.title('Top 10 Most Expensive and Least Expensive Stations (Based on One-Hot Encoding)')
plt.tight_layout()
plt.savefig('tokyo_rent_analysis_onehot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\nTop 10 Most Expensive Stations (by average cost per m²):")
print(station_avg_cost.nlargest(10, 'mean')[['station', 'mean', 'count']].to_string(
    index=False,
    float_format=lambda x: f"¥{x:,.0f}/m²" if isinstance(x, float) else str(x)
))

print("\nTop 10 Least Expensive Stations (by average cost per m²):")
print(station_avg_cost.nsmallest(10, 'mean')[['station', 'mean', 'count']].to_string(
    index=False,
    float_format=lambda x: f"¥{x:,.0f}/m²" if isinstance(x, float) else str(x)
))

# Print model scores
print("\nModel Scores:")
print(f"Polynomial Regression R²: {poly_model.score(X_poly, data['cost_per_sqm']):.3f}")
print(f"One-Hot Encoding R²: {onehot_model.score(station_dummies, data['cost_per_sqm']):.3f}")