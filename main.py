from fasthtml.common import *
import uvicorn

app = FastHTML()
rt = app.route

@rt("/")
def get():
  return Title("Starfish Exchange"), H1("Starfish Exchange"), P("Welcome to the Starfish Exchange!"), A("Click here to learn more", href="/about")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", default=8000)))