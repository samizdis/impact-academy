#!/bin/sh
set -e
pip install -r requirements.txt
export OPENAI_API_KEY='sk-xxx'
python main.py