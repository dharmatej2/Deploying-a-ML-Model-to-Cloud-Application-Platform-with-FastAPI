#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
exec uvicorn starter.main:app --host 0.0.0.0 --port $PORT