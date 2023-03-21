import argparse
import pandas as pd
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--columns",
        nargs='*',
        help="select columns to convert, select all if no any input"
    )
    parser.add_argument(
        "--no-header",
        action="store_false",
        help=""
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help=""
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    assert os.path.exists(args.file_path)

    df = pd.read_csv(args.file_path)
    
    if args.columns is not None:
        df = df[args.columns]
    
    df.to_csv(args.output_path, sep='\t', index=False, header=args.no_header)

if __name__ == "__main__":
    main()