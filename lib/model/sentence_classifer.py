import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class SentenceLSTMClassifier(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, lstm_hidden_dim, output_dim):
        """
        input_dim: Cümle embedding'lerinin boyutu.
        fc1_dim: FC1 katmanının çıkış boyutu.
        fc2_dim: FC2 katmanının çıkış boyutu.
        lstm_hidden_dim: LSTM katmanının hidden boyutu.
        output_dim: Sınıflandırma çıktısı (örneğin, sınıf sayısı).
        """
        super(SentenceLSTMClassifier, self).__init__()
        # LSTM, batch_first=True olarak tanımlandı.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)

    def forward(self, padded_x, lengths):
        packed_input = pack_padded_sequence(padded_x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out = h_n[-1]
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        out = self.fc3(x)
        return out


class SentenceTransClassifier(nn.Module):


    def __init__(self, input_dim, fc1_dim, fc2_dim, transformer_hidden_dim, output_dim):
        """
        input_dim: Cümle embedding'lerinin boyutu.
        fc1_dim: FC1 katmanının çıkış boyutu.
        fc2_dim: FC2 katmanının çıkış boyutu.
        transformer_hidden_dim: Transformer'ın (önceki LSTM hidden boyutu) boyutu.
                             (Not: Bu değer 4'e tam bölünebilir olmalıdır.)
        output_dim: Sınıflandırma çıktısı (örneğin, sınıf sayısı).
        """
        super(SentenceTransClassifier, self).__init__()
        # Girişin transformer'ın beklediği boyuta çekilmesi için projeksiyon
        self.input_proj = nn.Linear(input_dim, transformer_hidden_dim)

        # Transformer encoder katmanı; burada nhead=4 kullanılmıştır.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(transformer_hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)


    def forward(self, padded_x, lengths):
        """
        padded_x: [batch_size, seq_len, input_dim] boyutunda, padding uygulanmış giriş dizisi.
        lengths: Her örneğin geçerli (padding dışı) uzunluk bilgisi.
        """
        # Girişi istenen boyuta çekiyoruz.
        x_proj = self.input_proj(padded_x)  # [batch_size, seq_len, transformer_hidden_dim]

        # Transformer, padding olan yerleri mask'lemek için src_key_padding_mask bekler.
        # mask: True değeri, ilgili pozisyonun padding olduğunu belirtir.
        mask = torch.arange(x_proj.size(1), device=x_proj.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Transformer encoder katmanından geçirme
        transformer_out = self.transformer(x_proj, src_key_padding_mask=mask)

        # Her örnek için geçerli son token’ın temsili alınır.
        # lengths tensoründeki her değerin son geçerli indexi (length - 1) kullanılır.
        batch_size = transformer_out.size(0)
        seq_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, transformer_out.size(2))
        last_token = transformer_out.gather(dim=1, index=seq_indices).squeeze(1)

        # Son temsil fc katmanlarından geçirilir.
        x = self.fc1(last_token)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        out = self.fc3(x)
        return out