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
    

class AlignModel(torch.nn.Module):
    def __init__(self,
        whisper_model: whisper.Whisper,
        embed_dim: int=1280,
        hidden_dim: int=2048,
        hidden_layers: int=3,
        dropout: float=0.2,
        text_output_dim: int=10000,
        phoneme_output_dim: int=100,
        freeze_encoder: bool=False,
        device: str='cuda'
        ) -> None:
        super().__init__()
        self.whisper_model = whisper_model
        self.dtype = torch.float
        
        self.fc_text = nn.Sequential(
            BasicBlock(embed_dim, hidden_dim, dropout),
            *[BasicBlock(hidden_dim, hidden_dim, dropout) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, text_output_dim, dtype=self.dtype)
        )

        self.fc_phoneme = nn.Sequential(
            BasicBlock(embed_dim, hidden_dim, dropout),
            *[BasicBlock(hidden_dim, hidden_dim, dropout) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, phoneme_output_dim, dtype=self.dtype)
        )

        self.freeze_encoder = freeze_encoder
        self.device = device

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.whisper_model.embed_audio(x)
        else:
            embed = self.whisper_model.embed_audio(x)

        text_out = self.fc_text(embed)
        phoneme_out = self.fc_phoneme(embed)

        return text_out, phoneme_out