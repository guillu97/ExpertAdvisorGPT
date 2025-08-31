import os
import sys

# --- Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/ ---
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timezone, timedelta
from src.mt5_client import MT5Client
from pymt5linux import MetaTrader5 as mt5  # s'assure que le module répond

c = MT5Client(login=None, password=None, server=None, terminal_path=None)
c.connect()
df = c.fetch_ohlcv("EURUSD", "M5", datetime(2025,7,1,tzinfo=timezone.utc), datetime(2025,7,2,tzinfo=timezone.utc))
print(df.head()); print(df.shape)
c.shutdown()
