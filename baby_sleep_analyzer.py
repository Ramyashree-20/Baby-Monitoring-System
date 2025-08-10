# baby_sleep_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

def load_and_process_data(log_file="sleep_log.csv"):
    try:
        df = pd.read_csv(log_file)
        if df.empty: return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df['is_sleeping'] = np.where(df['detected_state'] == 'sleep', 1, 0)
        df['state_changed'] = df['is_sleeping'].diff() != 0
        df.iloc[0, df.columns.get_loc('state_changed')] = True
        return df[df['state_changed']].copy()
    except FileNotFoundError:
        return pd.DataFrame()

def create_timeline_and_prediction(event_df):
    if len(event_df) < 2: return None, None

    processed = []
    for i in range(len(event_df) - 1):
        duration = (event_df.iloc[i+1]['timestamp'] - event_df.iloc[i]['timestamp']).total_seconds() / 60
        state = 'sleep' if event_df.iloc[i]['is_sleeping'] == 1 else 'awake'
        processed.append({
            'start_time': event_df.iloc[i]['timestamp'],
            'end_time': event_df.iloc[i+1]['timestamp'],
            'duration_minutes': duration, 'state': state
        })

    if not processed: return None, None
    df = pd.DataFrame(processed)
    df['hour_of_day'] = df['start_time'].dt.hour
    df['previous_duration'] = df['duration_minutes'].shift(1).fillna(0)

    # --- Prediction Model ---
    training_df = df[df['state'] == 'awake']
    prediction = None
    if len(training_df) > 2 and df.iloc[-1]['state'] == 'awake':
        X = training_df[['hour_of_day', 'previous_duration']]
        y = training_df['duration_minutes']
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        last_awake = df.iloc[-1]
        features = pd.DataFrame([{'hour_of_day': last_awake['hour_of_day'], 'previous_duration': last_awake['previous_duration']}])
        predicted_duration = model.predict(features)[0]
        prediction = last_awake['start_time'] + timedelta(minutes=predicted_duration)

    return df, prediction

def generate_sleep_graph():
    """
    Main function to generate a clear, easy-to-understand line chart
    of the baby's sleep patterns.
    """
    event_data = load_and_process_data()
    if event_data.empty:
        return "<p>Not enough sleep data logged to generate a graph.</p>"

    processed_data, predicted_time = create_timeline_and_prediction(event_data)
    if processed_data is None:
         return "<p>Need at least two state changes in sleep_log.csv to generate a graph.</p>"

    fig = go.Figure()

    plot_x_values = []
    plot_y_values = []

    for index, row in processed_data.iterrows():
        plot_x_values.append(row['start_time'])
        plot_y_values.append(1 if row['state'] == 'sleep' else 0)

    # Add the final point to complete the last segment of the graph
    if not processed_data.empty:
        last_row = processed_data.iloc[-1]
        plot_x_values.append(last_row['end_time'])
        plot_y_values.append(1 if last_row['state'] == 'sleep' else 0)

    # --- NEW: Add the Step Chart Trace ---
    fig.add_trace(go.Scatter(
        x=plot_x_values,
        y=plot_y_values,
        mode='lines',
        line=dict(shape='hv', color='royalblue', width=2), # 'hv' creates the step shape
        fill='tozeroy',  # Fill the area between the line and the Y=0 axis
        fillcolor='rgba(173, 216, 230, 0.5)', # A light, transparent blue for the fill
        hoverinfo='none' # Disable the default hover text for a cleaner look
    ))

    # --- Prediction line remains the same ---
    if predicted_time:
        fig.add_vline(x=predicted_time, line_width=2, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=predicted_time, y=0.95, yref="paper",
            text=f"Predicted Sleep: {predicted_time.strftime('%H:%M')}",
            showarrow=True, arrowhead=1, ax=0, ay=-40
        )

    # --- NEW: Updated Layout for the Line Chart ---
    fig.update_layout(
        title_text='Baby Sleep Timeline',
        showlegend=False,
        height=350,
        xaxis_title="Time of Day",
        yaxis_title="State"
    )

    # Format the X-axis to show proper dates and times
    fig.update_xaxes(
        tickformat="%H:%M\n%b %d",
        ticklabelmode="period"
    )

    # Format the Y-axis to show "Awake" and "Sleep" instead of 0 and 1
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=['Awake', 'Sleep'],
        range=[-0.1, 1.1] # Add some padding to the top and bottom
    )

    return fig.to_html(full_html=False)