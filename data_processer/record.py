import os
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class Record:
    audio_path: str
    text_with_timestamp_path: str
    text: str
    language: str = 'zh'


def read_data_from_csv(data_path: str) -> List[Record]:
    assert os.path.exists(data_path)
    records = []
    df = pd.read_csv(data_path)
    
    for idx, row in df.iterrows():
        record = Record(audio_path=row['song_path'],
                        text_with_timestamp_path=row['lyric_with_char_timestamp_path'],
                        text=row['lyric'])
        records.append(record)

    return records