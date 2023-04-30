import os
import json
from typing import List, Optional
from dataclasses import dataclass
from ast import literal_eval
import pandas as pd

@dataclass
class Record:
    audio_path: str
    text: str
    phoneme_seq: Optional[str]=None
    timestamps: Optional[List[float]]=None

def phoneme_process(phoneme_seq: List[str]):
    for i in range(len(phoneme_seq)):
        if phoneme_seq[i] == 'w':
            phoneme_seq[i] = 'u'
        if phoneme_seq[i] == 'iou':
            phoneme_seq[i] = 'iu'
        if phoneme_seq[i] == 'io':
            phoneme_seq[i] = 'iu'
        if phoneme_seq[i] == 'un':
            phoneme_seq[i] = 'uen'
        if phoneme_seq[i] == 'uei':
            phoneme_seq[i] = 'ui'

    return phoneme_seq

def read_data_from_csv(data_path):
    assert os.path.exists(data_path)
    records = []
    df = pd.read_csv(data_path, converters={"phoneme_clean": literal_eval})
    
    for idx, row in df.iterrows():
        phoneme_seq = row['phoneme_clean']
        phoneme_seq = phoneme_process(phoneme_seq)

        record = Record(audio_path=row['song_path'],
                        text=row['lyric'],
                        phoneme_seq=phoneme_seq)
        records.append(record)

    return records

def read_data_from_json(
        data_path: str,
        has_phoneme: bool=True
    ) -> List[Record]:
    assert os.path.exists(data_path)
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    records = []
    for data in data_list:
        if has_phoneme:
            phoneme_seq = phoneme_process(data['phoneme_clean'])
            record = Record(audio_path=data['song_path'],
                            text=data['lyric'],
                            phoneme_seq=phoneme_seq)
        else:
            record = Record(audio_path=data['song_path'],
                            text=data['lyric'])
        records.append(record)

    return records