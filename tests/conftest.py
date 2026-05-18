"""Shared test fixtures + sys.path setup so `from core import ...` works."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
