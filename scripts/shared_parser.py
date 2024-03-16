import argparse
from config import TRAINING_DIR


def get_shared_parser(description: str = "Neural point based rendering") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("-s", "--scene", type=str, help="Name of the scene to render", default="staircase")
    parser.add_argument("-e", "--experiment", type=int, nargs="+", help="Training experiment", default=None)
    parser.add_argument("-t", "--training-dir", type=str, help="Training directory", default=TRAINING_DIR)
    return parser
