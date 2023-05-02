import torch
from torch import nn
import whisper

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(BasicBlock, self).__init__()

        self.dtype = torch.float

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim, dtype=self.dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 output_size,
                 num_layers: int=3,
                 dropout: float=0.2,
                 batch_first: bool=True, 
                 bidirectional: bool=True) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=batch_first,
                                bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + (bidirectional * hidden_size), output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.relu(out)
        out = self.fc(out)

        return out
    

class AlignModel(torch.nn.Module):
    def __init__(self,
        whisper_model: whisper.Whisper,
        embed_dim: int=1280,
        hidden_dim: int=1024,
        dropout: float=0.1,
        text_output_dim: int=10000,
        phoneme_output_dim: int=100,
        train_phoneme: bool=False,
        freeze_encoder: bool=False,
        device: str='cuda'
        ) -> None:
        super().__init__()
        self.whisper_model = whisper_model
        self.dtype = torch.float
        
        # Text Alignment
        self.text_rnn = RNN(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            output_size=text_output_dim,
                            dropout=dropout)

        # Phoneme LM
        self.train_phoneme = train_phoneme
        if train_phoneme:
            self.phoneme_rnn = RNN(input_size=embed_dim,
                                hidden_size=hidden_dim,
                                output_size=phoneme_output_dim,
                                dropout=dropout)
        else:
            self.phoneme_rnn = None



        self.freeze_encoder = freeze_encoder
        self.device = device

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.whisper_model.embed_audio(x)
        else:
            embed = self.whisper_model.embed_audio(x)

        text_logit = self.text_rnn(embed)
        if self.train_phoneme:
            phoneme_logit = self.phoneme_rnn(embed)
        else:
            phoneme_logit = None

        # TODO: Whisper Logit?

        return text_logit, phoneme_logit