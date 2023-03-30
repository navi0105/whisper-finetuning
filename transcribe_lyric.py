import argparse
import os
import json
import whisper
import torch
import pandas as pd
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from utils.timestamp_handler import TimestampHandler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="Data csv file for decode / transcription"
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=',',
        help="the separator between audio and transcription in data_file"
    )
    parser.add_argument(
        "--transcribe-with-timestamps",
        action='store_true',
        help=""
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default='song_path'
    )
    parser.add_argument(
        "--lyric-column",
        type=str,
        default='lyric'
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        help="Whisper model name or path"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help=(
            "Whether to perform X->X speech recognition ('transcribe')"
            "or X->English translation ('translate')"
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="Transcribe language"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=""
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result.csv"
    )
    parser.add_argument(
        "--output-sep",
        type=str,
        default=',',
        help=""
    )

    args = parser.parse_args()

    return args

def read_data(data_path: str, sep: str) -> list[list[str]]:
    assert os.path.exists(data_path)

    with open(data_path, 'r') as f:
        data = f.read().splitlines()
        for i in range(len(data)):
            data[i] = data[i].split(sep=sep)

    return data

def transcribe(model, data, task: str="transcribe", lang: str='zh'):
    audio_id = []
    lyric = []
    inference = []
    for row in tqdm(data, total=len(data), desc='Transcription'):
        audio_path = row[0]
        ground_truth = row[1]
        assert os.path.exists(audio_path)

        id = Path(audio_path).stem
        result = model.transcribe(task=task, 
                                  audio=audio_path,
                                  language=lang)
        
        audio_id.append(id)
        lyric.append(ground_truth)
        inference.append(result['text'])
    
    return {'audio_id': audio_id,
            'lyric': lyric,
            'inference': inference}

def transcribe_with_timestamps(
        model: whisper.Whisper, 
        data: pd.DataFrame,
        task: str="transcribe", 
        lang: str='zh'):
    transcribe_result = []
    for row in tqdm(data, total=len(data), desc='Transcription'):
        audio_path = row[0]
        ref_lyric_path = row[1]
        
        handler = TimestampHandler(file_path=ref_lyric_path)
        ref_text = handler.get_lyric()
        ref_result = handler.get_lyric_with_timestamp()

        pred_result = model.transcribe(task=task, 
                                       audio=audio_path,
                                       language=lang)
        
        pred_segments = pred_result['segments']

        transcribe_result.append({'ref_text': ref_text,
                                  'ref_timestamps': ref_result,
                                  'infernece_text': pred_result['text'],
                                  'inference_timestamps': pred_segments})
    return transcribe_result

def write_output(result, output_path, sep, with_timestmaps: bool=False):
    if with_timestmaps is not True:
        with open(output_path, 'w') as f:
            f.write(f'{sep}'.join(result.keys()) + '\n')
            for id, lyric, inference in tqdm(zip(result['audio_id'], result['lyric'], result['inference']), desc='Writing Output'):
                f.write(f'{sep}'.join([id, lyric, inference]) + '\n')
    else:
        assert os.path.splitext(output_path)[-1] == '.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)




def main():
    args = parse_args()
    device = args.device
    if device == 'cuda' and torch.cuda.is_available() != True:
        device = 'cpu'

    model = whisper.load_model(name=args.model, device=device)
    df = pd.read_csv(args.data_csv, sep=args.sep)
    data = df[[args.audio_column, args.lyric_column]].values

    if args.transcribe_with_timestamps:    
        transcribe_result = transcribe_with_timestamps(model=model,
                                                        data=data,
                                                        task=args.task,
                                                        lang=args.lang,
                                                        device=device)
    else:
        transcribe_result = transcribe(model=model,
                                        data=data,
                                        task=args.task,
                                        lang=args.lang)
    
    write_output(result=transcribe_result,
                 output_path=args.output,
                 sep=args.output_sep,
                 with_timestmaps=args.transcribe_with_timestamps)
    


if __name__ == "__main__":
    main()