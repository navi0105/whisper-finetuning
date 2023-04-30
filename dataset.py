from typing import List
from data_processor.record import Record
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES

from data_processor.record import read_data_from_json, read_data_from_csv

class OpencpopDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        tokenizer,
        phoneme_map: dict,
        has_phoneme_data: bool=False,
        fp16: bool=False
        ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.phoneme_map = phoneme_map
        self.fp16 = fp16

    def _calculate_mel(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        return mel

    def _encode_text(
        self,
        text: str
    ) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').view(-1)

    def _get_phoneme_token(
        self,
        phoneme_seq: List[str]
    ) -> torch.Tensor:
        for i in range(len(phoneme_seq)):
            phoneme_seq[i] = self.phoneme_map[phoneme_seq[i]]
        
        return torch.Tensor(phoneme_seq)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        mel = self._calculate_mel(record.audio_path)
        text_token = self._encode_text(record.text)
        phoneme_seq = self._get_phoneme_token(record.phoneme_seq)

        return (mel, text_token, phoneme_seq)
    
def collate_fn(data):
    x, y_text, y_phoneme = zip(*data)

    x = pad_sequence(x, batch_first=True, padding_value=0)
    y_text = pad_sequence(y_text, batch_first=True, padding_value=0)
    y_phoneme = pad_sequence(y_phoneme, batch_first=True, padding_value=-100)

    y_text[y_text == 0] = -100
    y_text[y_text == 102] = -100

    return x, y_text, y_phoneme

def get_dataloader(
    data_path: str,
    tokenizer,
    phoneme_map: dict,
    batch_size: int=1,
    fp16: bool=False,
    shuffle: bool=False
    ) -> DataLoader:
    assert os.path.exists(data_path)
    if os.path.splitext(data_path)[-1] == '.csv':
        records = read_data_from_csv(data_path)
    else:
        records = read_data_from_json(data_path)

    dataset = OpencpopDataset(records=records,
                              tokenizer=tokenizer,
                              phoneme_map=phoneme_map,
                              fp16=fp16)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )