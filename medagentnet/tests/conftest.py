"""Put the package root on sys.path so the modules import as they do at runtime."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)          # config/ is resolved relative to the package root
