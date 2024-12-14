#!/bin/bash

cd ..
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt