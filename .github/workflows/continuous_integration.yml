name: Python CI

on:
  push:
    branches:
      - master  # Trigger on pushes to the master branch
  pull_request:
    branches:
      - master  # Trigger on pull requests to the master branch

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8]
    steps:
      - name: Check out code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest

      - name: Run flake8
        run: |
          flake8
