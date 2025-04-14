#!/bin/bash
export PYTHONPATH=$(pwd)
exec uvicorn main:app --host 0.0.0.0 --port $PORT