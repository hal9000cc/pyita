#!/bin/bash
# Activate the pyita virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"
echo "Python: $(python --version)"
echo "pyita version: $(python -c 'import pyita; print(pyita.__version__)')"
