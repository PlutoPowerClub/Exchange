# Exchange

Web UI for the Exchange for energy buying and selling.

To install:

```bash
$ git clone
$ cd Exchange
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

To run locally:

```bash
$ python main.py --reload
```

To deploy:
    
```bash
brew install railway
railway login
railway up
```
