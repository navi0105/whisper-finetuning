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
                 dropout: float=0.1,
                 batch_first: bool=True, 
                 bidirectional: bool=True) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first=batch_first,
                                bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size + (bidirectional * hidden_size), output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
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
        freeze_encoder: bool=False,
        device: str='cuda'
        ) -> None:
        super().__init__()
        self.whisper_model = whisper_model
        self.dtype = torch.float
        
        # Text LM
        self.text_rnn = RNN(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            output_size=text_output_dim,
                            dropout=dropout)

        # Phoneme LM
        self.phoneme_rnn = RNN(input_size=embed_dim,
                               hidden_size=hidden_dim,
                               output_size=phoneme_output_dim,
                               dropout=dropout)



        self.freeze_encoder = freeze_encoder
        self.device = device

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.whisper_model.embed_audio(x)
        else:
            embed = self.whisper_model.embed_audio(x)

        text_out = self.text_rnn(embed)
        phoneme_out = self.phoneme_rnn(embed)

        return text_out, phoneme_out