import argparse
import json
from vib import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)

    args = parser.parse_args()
    with open(args.config_path) as fin:
        params = json.load(fin)

    run_experiment(params)


if __name__ == "__main__":
    main()
