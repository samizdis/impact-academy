#!/bin/sh
set -e
pip install -r requirements.txt
export OPENAI_API_KEY='sk-xx'
python main.py