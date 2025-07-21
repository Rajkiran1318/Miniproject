import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# Load dataset
try:
    df = pd.read_csv("rain_history.csv")  # Ensure this file exists in the same folder
except FileNotFoundError:
    print("Error: File 'rain_history.csv' not found.")
    exit()
except PermissionError:
    print(" Error: Permission denied for 'rain_history.csv'. Close the file if open.")
    exit()

# Check if required columns exist
required_columns = ['day', 'month', 'weekday', 'hour', 'rain']
if not all(col in df.columns for col in required_columns):
    print(f" Error: The dataset must contain these columns: {required_columns}")
    exit()

# Train model using all available years (2004–2025 assumed)
X = df[['day', 'month', 'weekday', 'hour']]
y = df['rain']

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n Model Accuracy: {accuracy * 100:.2f}%")

# Prediction function
def predict_rain(date_input, time_input):
    try:
        # Parse inputs
        dt = datetime.strptime(f"{date_input} {time_input}", "%Y-%m-%d %H:%M")
        day = dt.day
        month = dt.month
        weekday = dt.weekday()
        hour = dt.hour

        
        features = pd.DataFrame([[day, month, weekday, hour]], columns=['day', 'month', 'weekday', 'hour'])
        prediction = model.predict(features)

        if prediction[0] == 1:
            print(f"🌧 Yes, it may rain on {date_input} at {time_input}.")
        else:
            print(f" No, it likely won't rain on {date_input} at {time_input}.")
    except ValueError:
        print(" Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time.")


tomorrow = datetime.now() + timedelta(days=1)
predict_rain(tomorrow.strftime("%Y-%m-%d"), "14:00")


ask = input("\n Do you want to predict for a custom date/time? (yes/no): ").lower()
if ask == 'yes':
    d = input("Enter date (YYYY-MM-DD): ")
    t = input("Enter time (HH:MM): ")
    predict_rain(d, t)