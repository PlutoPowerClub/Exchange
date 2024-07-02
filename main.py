from fasthtml.common import *
import uvicorn

app = FastHTML()
rt = app.route

@rt("/")
def get():
  return Title("Starfish Exchange"), Main(
    Section(
      H1("Starfish Exchange"),
      P("Welcome to the Starfish Exchange!"),
      A("Click here to learn more", href="/about")
    ),
    Section(
      H2("Energy Prices"),
      P("Current energy prices are:"),
      Ul(
        Li("Solar: $0.02"),
        Li("Wind: $0.03"),  
        Li("Hydro: $0.04")
      )
    )
  )
    


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", default=8000)))
