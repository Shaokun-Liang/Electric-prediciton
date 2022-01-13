import torch
import pytorch_lightning as pl
from torchmetrics.functional import mean_absolute_percentage_error
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import inverse_norm
import torch.optim as optim
from config import config


class MultiHeadAttentionLayer(pl.LightningModule):

    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)

        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class GRUAttention(pl.LightningModule):

    def __init__(self):
        super(GRUAttention, self).__init__()

        self.GRU = torch.nn.GRU(input_size=config.num_features,
                                hidden_size=config.atten_hidden_size,
                                num_layers=config.atten_num_layer,
                                batch_first=True)
        self.pos_embedding = torch.nn.Embedding(24, config.atten_hidden_size)
        self.self_attention = MultiHeadAttentionLayer(config.atten_hidden_size, 4, 0.1)
        self.self_attn_layer_norm = torch.nn.LayerNorm(config.atten_hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.leaky_relu1 = torch.nn.LeakyReLU()
        self.fc_out1 = torch.nn.Linear(config.atten_hidden_size, config.atten_hidden_size // 2)
        self.leaky_relu2 = torch.nn.LeakyReLU()
        self.fc_out2 = torch.nn.Linear(config.atten_hidden_size // 2, 1)

    def make_mask(self, inputs):
        # input = [batch size, len, dim]

        batch_size = inputs.shape[0]
        length = inputs.shape[1]

        mask = torch.tril(torch.ones((length, length))).bool()

        # mask = [len, len]

        return mask.expand(batch_size, 1, length, length)

    def forward(self, src):
        output, _ = self.GRU(src)

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        output = output + self.pos_embedding(pos)
        output_mask = self.make_mask(output)

        _output, _ = self.self_attention(output, output, output)
        output = self.self_attn_layer_norm(output + self.dropout(_output))

        output = self.fc_out1(self.leaky_relu1(output))
        output = self.fc_out2(self.leaky_relu2(output))
        return output

    def training_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        loss = mean_absolute_percentage_error(y_pred, y_true.unsqueeze(1))
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = mean_absolute_percentage_error(y_pred, y_true.unsqueeze(1))
        self.log(name='val_mape', value=mape, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = mean_absolute_percentage_error(y_pred, y_true.unsqueeze(1))
        self.log(name='test_mape', value=mape)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=config.atten_lr)
        return opt

    def configure_callbacks(self):
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_mape',
                                                    mode='min',
                                                    patience=5,
                                                    check_on_train_epoch_end=False)
        checkpoint = ModelCheckpoint(monitor='val_mape',
                                     dirpath=config.mdoel_saving_dir,
                                     filename='GRUatten-{epoch}-{step}-{val_mape:.5f}')

        return [early_stopping, checkpoint]
