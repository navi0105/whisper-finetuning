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
    phoneme_seq: Optional[list]=None
    lyric_onset_offset: Optional[list]=None

    # timestamps: Optional[List[float]]=None

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

def opencpop_get_lyrics_word_onset_offset(data):
    onset_offset = []

    cur_onset_time = 0.0
    phoneme_accu_dur = 0.0
    for i in range(len(data['note_dur'])):

        phoneme_accu_dur = phoneme_accu_dur + data['phoneme_dur'][i]

        # print (cur_onset_time, phoneme_accu_dur, data['is_slur'][i], data['phoneme_dur'][i], data['note_dur'][i], data['phoneme'][i])

        if data['is_slur'][i] == 1:
            onset_offset[-1][-1] = onset_offset[-1][-1] + data['note_dur'][i]
            cur_onset_time = cur_onset_time + data['note_dur'][i]
            phoneme_accu_dur = 0.0

        if data['phoneme'][i] == 'SP' or data['phoneme'][i] == 'AP':
            cur_onset_time = cur_onset_time + data['note_dur'][i]
            phoneme_accu_dur = 0.0

        # End of a note
        # Add a small epsilon to avoid numirical issue
        if phoneme_accu_dur + 1e-6 >= data['note_dur'][i]:
            onset_offset.append([cur_onset_time, cur_onset_time + data['note_dur'][i]])
            cur_onset_time = cur_onset_time + data['note_dur'][i]
            phoneme_accu_dur = 0.0
            
    # print (data)
    # print (onset_offset)
    return onset_offset


def read_data_from_csv(data_path,
                       get_phoneme: bool=False):
    assert os.path.exists(data_path)
    records = []
    df = pd.read_csv(data_path, converters={"phoneme_clean": literal_eval})
    
    for idx, row in df.iterrows():
        if get_phoneme:
            phoneme_seq = row['phoneme_clean']
            phoneme_seq = phoneme_process(phoneme_seq)

            record = Record(audio_path=row['song_path'],
                            text=row['lyric'],
                            phoneme_seq=phoneme_seq)
        else:
            record = Record(audio_path=row['song_path'],
                            text=row['lyric'])
        records.append(record)

    return records

def read_data_from_json(
        data_path: str,
        get_phoneme: bool=False,
        only_alt: bool=False
    ) -> List[Record]:
    assert os.path.exists(data_path)
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    records = []
    for data in data_list:
        if get_phoneme and 'phoneme_clean' in data.keys():
            phoneme_seq = phoneme_process(data['phoneme_clean'])
            record = Record(audio_path=data['song_path'],
                            text=data['lyric'],
                            phoneme_seq=phoneme_seq)
        else:
            record = Record(audio_path=data['song_path'],
                            text=data['lyric'])
            
        if not only_alt:
            if 'lyric_onset_offset' in data:
                record.lyric_onset_offset = data['lyric_onset_offset']
            else:
                onset_offset = opencpop_get_lyrics_word_onset_offset(data)
                record.lyric_onset_offset = onset_offset

        records.append(record)

    return records