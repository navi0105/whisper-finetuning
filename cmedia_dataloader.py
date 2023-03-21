import re
from typing import List, Optional, Tuple
import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import Tokenizer

from utils.timestamp_handler import TimestampHandler

def load_data(data_path) -> dict:
    assert os.path.exists(data_path)
    df = pd.read_csv(data_path)
    
    data = df[['lyric_with_char_timestamp_path', 'song_path', 'lyric']].to_dict()
    return data

def get_lyric_with_timestamp(lyric_path) -> str:
    assert os.path.exists(lyric_path)

    with open(lyric_path, 'r') as f:
        rows = 



class AudioDataset(Dataset):
    def __init__(self,
                 data: dict,
                 tokenizer: Tokenizer,
                 fp16: bool=True) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.fp16 = fp16

        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH
        # timestamps tokens are from <|0.00|> to <|30.00|> with a step of 0.02
        self.timestamp_pattern = re.compile(r"(<\|[123]?[0-9]\.[0-9][0-9]\|>)")
        self.model_n_text_ctx = 448

