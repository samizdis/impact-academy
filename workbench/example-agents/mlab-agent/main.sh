#!/bin/sh
set -e
pip install -r requirements.txt
pip install --upgrade openai
export OPENAI_API_KEY='sk-proj-wIUiFL95SJ1tnWqmEA4GT3BlbkFJFSEiWRvQb3qfzJpgE975'
python main.py