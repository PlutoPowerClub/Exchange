from fasthtml.common import *
import uvicorn

app = FastHTMLWithLiveReload(hdrs=(
  # Script(src='https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js'),
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
      P("Here we empower your community to choose how you buy and sell energy in real time."),
      Select(
        Option("Simple Moving Average", value="sma"),
        Option("Exponential Moving Average", value="ema"),
        Option("Hold During Cloudy Weeks", value="cloud"),
        Option("Linear Regression", value="linreg"),
        Option("Random Forest", value="rf"),
        Option("Neural Network", value="nn"),
        Option("Other", value="other")
      ),
    ),
    Section(
      # Textarea(id="input", width="100%", height="800px", placeholder="Enter a message..."),
      Textarea(id="input", style="width: 60%; height: 400px;", placeholder="energy buy/sell algo loading...", content="if TODO"),
      Button("Save", onclick="save()")
    ),
  )
    


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", default=8000)))
