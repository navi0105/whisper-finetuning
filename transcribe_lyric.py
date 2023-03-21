import argparse
import os
import whisper
import torch
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Data file for decode / transcription, \
              the format of each line in file should be \
              \"<audio_path><sep><lyric>\" \
              e.g. audio_dir/audio_1.mp3\tspeech content in this audio \
              you can use --sep flag to change the separator between audio and transcription"
    )
    parser.add_argument(
        "--sep",
        type=str,
        default='\t',
        help="the separator between audio and transcription in data_file"
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

def write_output(result: dict, output_path, sep):
    with open(output_path, 'w') as f:
        f.write(f'{sep}'.join(result.keys()) + '\n')
        for id, lyric, inference in tqdm(zip(result['audio_id'], result['lyric'], result['inference']), desc='Writing Output'):
            f.write(f'{sep}'.join([id, lyric, inference]) + '\n')

def main():
    args = parse_args()
    device = args.device
    if device == 'cuda' and torch.cuda.is_available() != True:
        device = 'cpu'

    model = whisper.load_model(name=args.model, device=device)
    data = read_data(data_path=args.data_file,
                     sep=args.sep)
    
    transcribe_result = transcribe(model=model,
                                    data=data,
                                    task=args.task,
                                    lang=args.lang)
    
    write_output(result=transcribe_result,
                 output_path=args.output,
                 sep=args.output_sep)
    


if __name__ == "__main__":
    main()