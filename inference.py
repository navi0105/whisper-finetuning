import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List
from tqdm import tqdm
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from torchaudio.models.decoder import ctc_decoder

from module.align_model import AlignModel
from dataset import get_dataloader

os.environ["TOKENIZERS_PARALLELISM"]="false"

whisper_dim = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

phoneme_list = ['blank', 'r', 'ang', 'uo', 'z', 'ai', 'ch', 'i', 'k', 'ou', 'q', 't', 'ian', 'd', 'e', 'j',
                've', 'ong', 'x', 'in', 'iao', 'a', 'ie', 'h', 'ui', 'sh', 'c', 'b', 'ei', 'uan', 'ing', 'm', 'n', 'l'
                , 'u', 'iang', 'zh', 'eng', 'ao', 'en', 'g', 'uang', 'an', 'uai', 'v', 'p', 'f', 'iong', 'uen', 'ua'
                , 'van', 'ia', 'vn', 's', 'o', 'er', 'iu', 'ueng', 'y']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--test-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--whisper-model',
        type=str,
        default='large'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-chinese'
    )
    parser.add_argument(
        '--state-dict',
        type=str,
        default=None
    )
    parser.add_argument(
        '--config-json',
        type=str,
        default=None
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--result-file',
        type=str,
        default='result.csv'
    )

    args = parser.parse_args()
    return args

def ctc_text_handle(
        results: List[str],
        blank_token: str
    ) -> List[str]:
    processed_result = []
    for result in results:
        result = result.split(' ')
        curr_text = ''
        curr_char = ''
        for char in result:
            if char != curr_char:
                curr_text += char
                curr_char = char
        # remove blank token
        curr_text = curr_text.replace(blank_token, '')
        processed_result.append(curr_text)

    return processed_result
        

def text_decoder(logits: torch.tensor, tokenizer):
    logits = torch.argmax(logits, dim=-1)
    decoded_texts = tokenizer.batch_decode(logits)

    return ctc_text_handle(decoded_texts, '[PAD]')\
    
def phoneme_decoder(
        logits: torch.tensor,
        idx2phoneme,
        ctc_decoder
    ) -> List[List[str]]:
    mapping_fn = np.vectorize(idx2phoneme.get)

    decoded_results = ctc_decoder(logits.cpu())
    decoded_phoneme_seqs = []
    for result in decoded_results:
        phoneme_seq = []
        tokens = result[0].tokens.numpy()
        tokens = tokens[tokens < len(idx2phoneme) - 2]

        for token in tokens:
            phoneme_seq.append(idx2phoneme[token])

        decoded_phoneme_seqs.append(phoneme_seq)

    return decoded_phoneme_seqs
    


def main():
    args = parse_args()
    
    if args.config_json is not None:
        assert os.path.exists(args.config_json)
        with open(args.config_json, 'r') as f:
            train_config = json.load(f)
    else:
        train_config = None
    
    device = args.device
    if 'cuda' in device and torch.cuda.is_available() != True:
        device = 'cpu'

    whisper_model_name = args.whisper_model
    tokenizer_name = args.tokenizer
    freeze_encoder = False
    if train_config is not None:
        whisper_model_name = train_config['whisper_model']
        tokenizer_name = train_config['tokenizer']
        freeze_encoder = train_config['freeze_encoder']    

    phoneme2idx = {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}
    idx2phoneme = {idx: phoneme for idx, phoneme in enumerate(phoneme_list)}

    idx2phoneme[len(idx2phoneme)] = '|'
    idx2phoneme[len(idx2phoneme)] = '<unk>'

    whisper_model = whisper.load_model(whisper_model_name,
                                       device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = AlignModel(whisper_model=whisper_model,
                       embed_dim=whisper_dim[whisper_model_name],
                       text_output_dim=len(tokenizer),
                       phoneme_output_dim=len(phoneme2idx),
                       freeze_encoder=freeze_encoder,
                       device=device)
    
    # Load Trained State Dict
    if args.state_dict is not None:
        assert os.path.exists(args.state_dict)
        state_dict = torch.load(args.state_dict)['model_state_dict']
        
        model.load_state_dict(state_dict=state_dict)
    
    test_dataloader = get_dataloader(data_path=args.test_data,
                                     tokenizer=tokenizer,
                                     phoneme_map=phoneme2idx,
                                     batch_size=args.test_batch_size,
                                     shuffle=False)
    
    beam_decoder = ctc_decoder(lexicon=None,
                                tokens=phoneme_list + ['|'] + ['<unk>'],
                                blank_token='blank')

    model.to(device)
    model.eval()

    predict_text = []
    predict_phoneme = []
    target_text = []
    target_phoneme = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=(len(test_dataloader))):
            mel, tgt_text, tgt_phoneme = batch
            text_logits, phoneme_logits = model(mel.to(device))

            # Text
            pred_text = text_decoder(text_logits, tokenizer)
            predict_text.extend(pred_text)
            
            tgt_text[tgt_text == -100] = 0
            tgt_text = tokenizer.batch_decode(tgt_text, skip_special_tokens=True)
            for tgt in tgt_text:
                target_text.append(tgt.replace(' ', ''))

            # Phoneme
            pred_phoneme = phoneme_decoder(phoneme_logits, idx2phoneme, beam_decoder)
            predict_phoneme.extend(pred_phoneme)

            for tgt in tgt_phoneme:
                tgt = tgt
                tgt = np.delete(tgt, tgt == -100)
                tgt = np.vectorize(idx2phoneme.get)(tgt).tolist()
                target_phoneme.append(tgt)


    pd.DataFrame({'lyric': target_text,
                  'inference_lyric': predict_text,
                  'phoneme_seq': target_phoneme,
                  'inference_phoneme_seq': predict_phoneme}).to_csv(args.result_file, index=False)
    
    

if __name__ == "__main__":
    main()
