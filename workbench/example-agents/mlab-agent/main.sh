#!/bin/sh
set -e
pip install -r requirements.txt
pip install --upgrade openai
export OPENAI_API_KEY=''
python main.py