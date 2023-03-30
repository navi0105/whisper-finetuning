import json
import os
import argparse
from tqdm import tqdm
from utils.MAE import MAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--result-file",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default='.json'
    )
    args = parser.parse_args()
    return args

def timestamp_extractor(file_path: str, ext: str):
    assert os.path.splitext(file_path)[-1] == ext
    if ext == '.json':
        with open(file_path, 'r') as f:
            result_data = json.load(f)
    
    pred_timestamps = []
    ref_timestamps = []
    not_match_count = 0
    for data in tqdm(result_data):
        if len(data['infernece_text']) == len(data['ref_text']) \
        and len(data['inference_timestamps']) == len(data['ref_timestamps']):
            pred = []
            ref = []
            # infernece timestamps
            for info in data['inference_timestamps']:
                pred.append(info['start'])
                pred.append(info['end'])
            # reference timestamps
            for info in data['ref_timestamps']:
                ref.append(info['start'])
                ref.append(info['end'])

            pred_timestamps.append(pred)
            ref_timestamps.append(ref)
        else:
            not_match_count += 1

    print(f"Not match data count: {not_match_count} / {len(result_data)}")
    return pred_timestamps, ref_timestamps

def main():
    args = parse_args()
    prediction, reference = timestamp_extractor(args.result_file, args.data_format)

    mae = 0.0
    for pred, ref in tqdm(zip(prediction, reference), total=len(reference)):
        mae += MAE(pred=pred, ref=ref)
    mae = mae / len(reference)
    print(f"Average MAE of all matched data: {mae:.4f}(s)")

if __name__ == "__main__":
    main()