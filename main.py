from fasthtml.common import *
import json
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
  }
]

@rt("/")
def get():
  return Title("Starfish Exchange"), Main(
    Section(
      H1("Starfish Exchange"),
    ),
    Section(
      H2("Energy Prices in Real Time"),
      Div(id="chart", width="100%", height="500px")
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
        Option("Neural Network", value="nn"),
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
