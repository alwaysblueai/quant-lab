#!/usr/bin/env python3
"""CLI entrypoint: run a factor experiment from the command line.

Usage:
    uv run python scripts/run_experiment.py --input-path data/raw/prices.csv \
        --factor momentum --label-horizon 5 --quantiles 5
"""
import sys

from alpha_lab.cli import main

sys.exit(main())
