import os
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Record:
    audio_path: str
    text_with_timestamp_path: str
    text: str
    language: str = 'zh'


def read_data_from_csv(data_path: str,
                       audio_path_col: str='song_path',
                       text_with_timestamp_path_col: Optional[str]='lyric_with_char_timestamp_path',
                       text_col: str='lyric') -> List[Record]:
    assert os.path.exists(data_path)
    records = []
    df = pd.read_csv(data_path)
    
    for idx, row in df.iterrows():
        record = Record(audio_path=row[audio_path_col],
                        text_with_timestamp_path=row[text_with_timestamp_path_col] if text_with_timestamp_path_col is not None else '',
                        text=row[text_col])
        records.append(record)

    return records
