import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load your dataset, replace paths with yours. support both sql db formant and csv format
data = pd.read_csv('/Users/ethan/python_learn/ML_pratices/properties.csv')

# Identify numeric and categorical columns globally
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Data preprocessing
# enter the feature you want to anylize, i choose "cost". you should ctrl+f and replaces all 'cost' with your target feature
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Handle missing values
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create dummy variables for categorical features
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col != 'cost'])
    
    return df


# Scale numeric features
processed_data = preprocess_data(data)

scaler = StandardScaler()
numeric_cols = [col for col in numeric_features if col != 'cost']
processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])

# Split into features and target
y = processed_data['cost']
X = processed_data.drop('cost', axis=1)

# Split into training and testing sets

# Feature selection using RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
selector = SelectFromModel(model, prefit=False)
selector.fit(X_train, y_train)
feature_mask = selector.get_support()
selected_features = X.columns[feature_mask].tolist()

# Select features based on importance
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
# Train the model with selected features
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_train_selected, y_train)

# Make predictions

y_pred = final_model.predict(X_test_selected)
# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = final_model.score(X_test_selected, y_test)

print("\nModel Performance Metrics:")
print(f'Mean Absolute Error: ¥{mae:,.0f}')
print(f'Root Mean Squared Error: ¥{rmse:,.0f}')
print(f'R² Score: {r2:.3f}')

# Visualize feature importance
def plot_feature_importance_with_direction(model, X, y, feature_names, top_n=10):
    feature_importance = model.feature_importances_
    
    correlations = pd.DataFrame(X, columns=feature_names).corrwith(pd.Series(y))
    
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'correlation': correlations
    })
    
    feature_imp_df = feature_imp_df.sort_values('importance', ascending=True).tail(top_n)
    
    colors = ['red' if x < 0 else 'green' for x in feature_imp_df['correlation']]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feature_imp_df)), feature_imp_df['importance'])
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(feature_imp_df)), feature_imp_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features\nGreen = Positive Impact, Red = Negative Impact')
    
    for i, (importance, correlation, feature) in enumerate(zip(
            feature_imp_df['importance'], 
            feature_imp_df['correlation'], 
            feature_imp_df['feature'])):
        plt.text(importance, i, f' r={correlation:.2f}', va='center')
    
    plt.tight_layout()
plt.show()

selected_features = X.columns[selector.get_support()]
X_selected = X[selected_features]
plot_feature_importance_with_direction(final_model, X_selected, y, np.array(selected_features))

correlations = pd.DataFrame(X_selected, columns=selected_features).corrwith(pd.Series(y)).sort_values(ascending=False)
print("\nFeature Correlations with Target (cost):")
for feature, corr in correlations.items():
    direction = "positively" if corr > 0 else "negatively"
    print(f"{feature}: {direction} correlated ({corr:.3f})")


##### make a plot of the correlation
plt.figure(figsize=(12, 8))
feature_imp_df = pd.DataFrame({
    'feature': selected_features,
    'importance': final_model.feature_importances_,
    'correlation': correlations
})

# Sort by importance
feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)

# Create bubble chart
plt.scatter(feature_imp_df['importance'], 
            feature_imp_df['correlation'], 
            s=abs(feature_imp_df['correlation']) * 1000,  # Size of bubble
            alpha=0.6,
            c=feature_imp_df['correlation'],  # Color based on correlation
            cmap='coolwarm')  # Red for positive, Blue for negative correlations

plt.xlabel('Feature Importance')
plt.ylabel('Correlation with Cost')
plt.title('Feature Importance vs Correlation with Cost')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Correlation Strength')

# Annotate points with feature names
for idx, row in feature_imp_df.iterrows():
    plt.annotate(row['feature'], 
                 (row['importance'], row['correlation']),
                 xytext=(5, 5), 
                 textcoords='offset points',
                 fontsize=8,
                 alpha=0.8)

plt.tight_layout()
plt.show()
