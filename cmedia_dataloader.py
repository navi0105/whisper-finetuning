import re
from typing import List
import os

import whisper
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import Tokenizer

from utils.timestamp_handler import TimestampHandler
from data_processer.record import Record
from data_processer.record import read_data_from_csv

class CmediaDataset(Dataset):
    def __init__(
            self,
            records: List[Record],
            tokenizer: Tokenizer,
            language: str='zh',
            fp16: bool=True,
            no_timestamps: bool=False
    ) -> None:
        # record -> audio_path, text_path_with_char_timestamp
        self.records = records
        self.tokenizer = tokenizer
        self.language = language
        self.fp16 = fp16
        self.no_timestamps = no_timestamps
    
    def _encode_text_with_timestamps(
            self,
            text_path: str
    ) -> List[int]:
        handler = TimestampHandler(text_path)
        text_with_timestamp = handler.get_lyric_with_timestamp()
        tokens = []
        for char_info in text_with_timestamp:
            if char_info['start'] < 0 or char_info['start'] > 30:
                raise ValueError(f"Invalid timestamp: {char_info['start']}")
            if char_info['end'] < 0 or char_info['end'] > 30:
                raise ValueError(f"Invalid timestamp: {char_info['end']}")
            
            # start timestamp
            start_token = self.tokenizer.timestamp_begin + char_info['start'] * 100 // 2
            tokens.append(start_token)
            # char token
            tokens.extend(self.tokenizer.encode(char_info['char']))
            # end timestamp
            end_token = self.tokenizer.timestamp_begin + char_info['end'] * 100 // 2
            tokens.append(end_token)

        return tokens
    

    def _construct_decoder_output(
            self,
            special_tokens: List[int],
            text_tokens: List[int]
    ) -> List[int]:
        decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        return decoder_output

    def _calculate_mel(
            self,
            audio_path: str,
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        return mel



    def _get_special_tokens(
            self, 
            is_text_empty: bool, 
            language: str, 
            no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens

    def _get_text_tokens(
            self,
            record: Record,
            no_timestmaps: bool
    ) -> List[int]:
        if no_timestmaps == False:
            text_tokens = self._encode_text_with_timestamps(record.text_with_timestamp_path)
        else:
            text_tokens = self.tokenizer.encode(record.text)

        return text_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        no_timestamps = self.no_timestamps
        
        text_tokens = self._get_text_tokens(record, no_timestamps)
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, no_timestamps)

        decoder_input = special_tokens + text_tokens
        decoder_output = self._construct_decoder_output(special_tokens, text_tokens)
        mel = self._calculate_mel(record.audio_path)

        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long)
        )


def collate_fn(data):
    x, y_in, y_out = zip(*data)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
    return x, y_in, y_out

def get_dataloader(
    csv: str,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    language: str = 'zh',
    fp16: bool = True,
    no_timestamps: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    records = read_data_from_csv(csv)
    dataset = CmediaDataset(
        records,
        tokenizer,
        language=language,
        fp16=fp16,
        no_timestamps=no_timestamps
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )