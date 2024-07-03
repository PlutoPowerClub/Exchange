algos = [
  {
    "id": "cloud",
    "name": "Hold During Extended Cloud Cover",
    "code": """def seller(current_consumption, current_generation, latitude, longitude, days):
    weather = __weather_rating(__weather(latitude, longitude, days))
    battery_status = __battery_capacity()
    # Trade cautiously during bad weather
    if current_consumption < current_generation:
        trade = 1
        hint = ''
    else:
        trade = 0
        hint = 'Usage is below generation capacity.'
    if trade == 1:
        if weather == 'bad' and battery_status == 'low':
            trade = 0
            hint = "Charging batteries, not selling."
        if weather == 'good' and battery_status == 'high':
            trade = 1
            hint = 'Selling energy at 15p/kWh.'
        if weather == 'bad' and battery_status == 'high':
            trade = 1
            hint = 'Selling energy at 15p/kWh.'
    response = {"trade":trade, "hint": hint}
    return response""",
    "description": "Buy/sell only when it is mostly sunny, to ensure our community has energy to use for our needs.",
    "params": []
  },
  {
    "id": "sma",
    "name": "Simple Moving Average",
    "code": "def sma(data, window=5):\n  return data.rolling(window=window).mean()",
    "description": "Simple Moving Average is the average of the last n periods.",
    "params": [
      {"name": "window", "type": "int", "default": 5, "description": "The number of periods to average."}
    ]
  },
  {
    "id": "ema",
    "name": "Exponential Moving Average",
    "code": "def ema(data, window=5):\n  return data.ewm(span=window, adjust=False).mean()",
    "description": "Exponential Moving Average is the weighted average of the last n periods.",
    "params": [
      {"name": "window", "type": "int", "default": 5, "description": "The number of periods to average."}
    ]
  },
  {
    "id": "linreg",
    "name": "Linear Regression",
    "code": "def linreg(data):\n  return data",
    "description": "Use linear regression to predict future prices.",
    "params": []
  },
  {
    "id": "rf",
    "name": "Random Forest",
    "code": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load historical data
data = pd.read_csv('solar_weather_power_data.csv')

# Features: weather data (temperature, cloud cover, etc.)
X = data[['temperature', 'cloud_cover', 'wind_speed', 'humidity']]
# Target: solar power generation
y = data['power_generation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Function to make buy/sell decision
def decide_action(weather_data):
    predicted_power = model.predict([weather_data])[0]
    if predicted_power > threshold:
        return 'sell'
    else:
        return 'buy'""",
    "description": "Use a random forest to predict future prices.",
    "params": []
  },
  {
    "id": "nnpytorch",
    "name": "Neural Network (PyTorch)",
    "code": """import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('solar_weather_power_data.csv')
X = data[['temperature', 'cloud_cover', 'wind_speed', 'humidity', 'time_of_day', 'day_of_year']].values
y = data['power_generation'].values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# Define the model
class SolarNet(nn.Module):
    def __init__(self, input_dim):
        super(SolarNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Initialize model, loss, and optimizer
model = SolarNet(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Prediction function
def predict(weather_data):
    model.eval()
    with torch.no_grad():
        scaled_data = torch.FloatTensor(scaler.transform([weather_data]))
        return model(scaled_data).item()

# Decision function
def decide_action(weather_data):
    predicted_power = predict(weather_data)
    return 'sell' if predicted_power > threshold else 'buy'
""",
    "description": "Use a neural network to predict future prices.",
    "params": []
  }
]
