from algos import algos
from fasthtml.common import *
import json
import uvicorn

app = FastHTMLWithLiveReload(hdrs=(
  # Script(src='https://cdn.jsdelivr.net/npm/jquery'),
  Script(src='https://cdn.jsdelivr.net/npm/apexcharts'),
  Script(src='AnnotatedChart.js'),
  )
)
rt = app.route

# Static files
@rt("/{fname:path}.{ext:static}")
async def get(fname:str, ext:str): return FileResponse(f'{fname}.{ext}')


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
      P("Choose exactly how you buy and sell energy in real time."),
      Select(
        Option("Hold During Extended Cloud Cover", value="cloud"),
        Option("Simple Moving Average", value="sma"),
        Option("Exponential Moving Average", value="ema"),
        Option("Linear Regression", value="linreg"),
        Option("Random Forest", value="rf"),
        Option("Neural Network (PyTorch)", value="nnpytorch"),
        Option("Other", value="other"),
        id="algo-select",
        style="width: 100%;",
        onchange="loadAlgo()",
      ),
    ),
    Section(
      Textarea(algos[0]["code"], id="algo-input", style="width: 100%; height: 400px;", placeholder="energy buy/sell algo loading..."),
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
