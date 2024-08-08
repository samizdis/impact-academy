#!/bin/sh
set -e
pip install -r requirements.txt
export OPENAI_API_KEY='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'
python main.py