# calculate_prediction_accuracy.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def create_feature_dataframe(log_file="sleep_log.csv", min_duration_minutes=1):
    """
    A new, more robust function to load, clean, and engineer features
    from the raw log file.
    """
    try:
        df = pd.read_csv(log_file)
        if len(df) < 2: return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # 1. Identify only the points where the state actually changes
        df['state_change'] = df['detected_state'] != df['detected_state'].shift(1)
        events = df[df['state_change'] == True].copy()
        
        # 2. Calculate the duration of each state period
        events['duration_minutes'] = events['timestamp'].diff().dt.total_seconds().shift(-1) / 60
        
        # 3. Filter out short, noisy events
        events = events[events['duration_minutes'] >= min_duration_minutes]
        
        # 4. Separate into sleep and awake periods for feature engineering
        awake_periods = events[events['detected_state'] == 'awake'].copy()
        sleep_periods = events[events['detected_state'] == 'sleep'].copy()
        
        if awake_periods.empty or sleep_periods.empty:
            return None

        # 5. Create the features for each 'awake' period
        feature_list = []
        for index, awake_row in awake_periods.iterrows():
            # Find all sleep periods that happened BEFORE this awake period
            past_sleeps = sleep_periods[sleep_periods['timestamp'] < awake_row['timestamp']]
            
            if not past_sleeps.empty:
                # Feature 1: The average duration of all past naps
                avg_sleep_duration = past_sleeps['duration_minutes'].mean()
                
                # Feature 2: The hour of the day
                hour_of_day = awake_row['timestamp'].hour
                
                # The Target: The duration of this awake period
                target_duration = awake_row['duration_minutes']
                
                feature_list.append({
                    'hour_of_day': hour_of_day,
                    'avg_past_sleep_duration': avg_sleep_duration,
                    'duration_minutes': target_duration
                })
                
        if not feature_list: return None
        
        # Return a clean DataFrame ready for modeling
        return pd.DataFrame(feature_list).dropna()

    except FileNotFoundError:
        print(f"ERROR: Log file '{log_file}' not found.")
        return None


def evaluate_prediction_models():
    """
    Trains and evaluates models using the new, robust feature creation.
    """
    # Use the new function to get the data
    training_df = create_feature_dataframe()

    if training_df is None or len(training_df) < 5:
        print("Not enough valid 'awake' periods found in the data to perform evaluation.")
        print("Please ensure your log has several complete sleep/awake cycles of at least 1 minute each.")
        return

    X = training_df[['hour_of_day', 'avg_past_sleep_duration']]
    y = training_df['duration_minutes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if len(X_test) == 0:
        print("Test set is empty after splitting. Please add more sleep/wake cycles to your data.")
        return

    models = {
        'Our Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Simple Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        results[name] = mae

    print("\n--- Sleep Prediction Model Performance Report ---")
    print("Metric: Mean Absolute Error (Lower is Better)\n")
    for name, score in sorted(results.items(), key=lambda item: item[1]):
        print(f"- {name}: {score:.2f} minutes")

if __name__ == '__main__':
    evaluate_prediction_models()