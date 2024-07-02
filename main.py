from fasthtml.common import *
import httpx
import json
import pandas as pd
import uvicorn

app = FastHTMLWithLiveReload(hdrs=(
  Script(src='https://cdn.jsdelivr.net/npm/apexcharts'),
  Script(src='AnnotatedChart.js'),
  )
)
rt = app.route

# Static files
@rt("/{fname:path}.{ext:static}")
async def get(fname:str, ext:str): return FileResponse(f'{fname}.{ext}')

algos = [
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
    "id": "cloud",
    "name": "Hold During Cloudy Weeks",
    "code": "def cloud(data):\n  return data",
    "description": "Like SMA, but only buy/sell during cloudy weeks.",
    "params": []
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

# Get data from Octopus API, and reformat for ApexCharts
url = "https://api.octopus.energy/v1/products/GO-VAR-22-10-14/electricity-tariffs/E-1R-GO-VAR-22-10-14-A/standard-unit-rates/?period_from=2024-06-30T00:00Z&period_to=2024-07-02T12"
response = httpx.get(url)
data = response.json()
df = pd.DataFrame(data['results'])
series = {
    "monthDataSeries1": {
        "prices": [],
        "dates": []
    }
}
for result in data["results"]:
    value_inc_vat = result["value_inc_vat"]
    valid_from = datetime.fromisoformat(result["valid_from"][:-1])  # Remove the 'Z' from the end
    valid_from_str = valid_from.strftime("%d %b %Y")

    series["monthDataSeries1"]["prices"].append(value_inc_vat)
    series["monthDataSeries1"]["dates"].append(valid_from_str)
series_json = json.dumps(series)

@rt("/")
def get():
  return Title("Starfish Exchange"), Main(
    Section(
      H1("Starfish Exchange"),
    ),
    Section(
      H2("Energy Prices in Real Time"),
      Div(id="chart", width="100%", height="500px")
      # TODO pass series_json into AnnotatedChart.js
    ),
    Section(
      H2("Algorithm"),
      P("Here we empower your community to choose how you buy and sell energy in real time."),
      Select(
        Option("Simple Moving Average", value="sma"),
        Option("Exponential Moving Average", value="ema"),
        Option("Hold During Cloudy Weeks", value="cloud"),
        Option("Linear Regression", value="linreg"),
        Option("Random Forest", value="rf"),
        Option("Neural Network (PyTorch)", value="nnpytorch"),
        Option("Other", value="other"),
        id="algo-select",
        onchange="loadAlgo()",
      ),
    ),
    Section(
      Textarea(algos[0]["code"], id="algo-input", style="width: 60%; height: 400px;", placeholder="energy buy/sell algo loading..."),
      Button("Save", onclick="save()")
    ),
    Script(f"""
      const algos = {json.dumps(algos)};
      function loadAlgo() {{
        const algoId = document.getElementById("algo-select").value;
        const algo = algos.find(a => a.id === algoId);
        if (algo) {{
          document.getElementById("algo-input").value = algo.code;
        }}
      }}
    """)
  )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", default=8000)))
