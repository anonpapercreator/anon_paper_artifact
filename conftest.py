"""
conftest.py — pytest path bootstrap
====================================
Adds the project root to sys.path before pytest collects any tests.
This means 'python -m pytest' from anywhere inside the project will
correctly resolve 'core', 'components', 'workload', etc.

No imports here — just the path insertion.
"""
import sys
import os

# Insert the directory containing this file (the project root) at position 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
