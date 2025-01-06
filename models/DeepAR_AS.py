import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class DeepAR(nn.Module):
    """
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Paper link: https://arxiv.org/abs/1704.04110
    """

    def __init__(self, configs, device):
        super(DeepAR, self).__init__()
        self.is_train = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_size = self.seq_len + self.pred_len
        self.time_num = configs.time_num
        self.time_cat = configs.time_cat
        self.tgt_num = configs.tgt_num
        self.meta_num = configs.meta_num
        self.meta_cat = configs.meta_cat
        self.c_out = self.tgt_num
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention
        self.device = device

        self.hidden_size = 128
        self.num_layers = 2

        # Time covariates size based on frequency
        freq_to_time_cov = {
            'daily': 2,
            'h': 1,
        }
        # self.time_cov_size = freq_to_time_cov[configs.freq]
        self.time_cov_size = configs.time_cov_size

        # Embedding layers
        self.time_embed_map = ModuleList([
            nn.Embedding(configs.time_num_class[i], configs.time_cat_embed[i])
            for i in range(self.time_cat)
        ])
        self.meta_embed_map = ModuleList([
            nn.Embedding(configs.meta_num_class[i], configs.meta_cat_embed[i])
            for i in range(self.meta_cat)
        ])

        # Calculate input size for LSTM - with debugging
        time_feat_size = self.time_num + sum(configs.time_cat_embed)
        meta_feat_size = self.meta_num + sum(configs.meta_cat_embed)
        
        # print(f"Target size (prev_y): {self.tgt_num}")
        # print(f"Time features size: {time_feat_size}")
        # print(f"Time covariates size: {self.time_cov_size}")
        # print(f"Meta features size: {meta_feat_size}")
        
        self.input_size = (
            self.tgt_num  # previous target values
            + time_feat_size  # time features
            + self.time_cov_size  # time covariates
            + meta_feat_size  # meta features
        )
        # print(f"Total input size calculated: {self.input_size}")

        # Encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=configs.dropout,
            batch_first=True
        )
        
        # Decoder
        self.mean_layer = nn.Linear(self.hidden_size, self.c_out)
        self.sigma_layer = nn.Linear(self.hidden_size, self.c_out)  

    def _get_embeddings(self, given_enc, meta_x):
        """Helper function to compute embeddings for categorical features"""
        b_sz = given_enc.size(0)
        
        # Embed time categorical features
        given_enc_cat_list = [
            self.time_embed_map[i](given_enc[:, :, -self.time_cat + i].long())
            for i in range(self.time_cat)
        ]
        given_enc_embed = torch.cat(
            [given_enc[:, :, :-self.time_cat]] + given_enc_cat_list, 
            dim=-1
        )
        
        # Embed meta categorical features
        if meta_x is not None:
            meta_x_cat_list = [
                self.meta_embed_map[i](meta_x[:, -self.meta_cat + i].long())
                for i in range(self.meta_cat)
            ]
            meta_x_embed = torch.cat(
                [meta_x[:, :-self.meta_cat]] + meta_x_cat_list,
                dim=-1
            )
        else:
            meta_x_embed = torch.zeros(b_sz, self.meta_num + sum(configs.meta_cat_embed)).to(self.device)
            
        return given_enc_embed, meta_x_embed

    def forecast(self, given_enc, x_enc, x_mark_enc, meta_x, output_attention=False):
        batch_size = x_enc.size(0)
        
        # Initialize hidden states
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        hidden = (h, c)

        # Get embeddings
        given_enc_embed, meta_x_embed = self._get_embeddings(given_enc, meta_x)
        

        # Window-wise normalization if configured
        norm_mean = None
        norm_std = None
        if self.use_norm:
            norm_mean = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - norm_mean
            norm_std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / norm_std

        # means = []
        means = torch.zeros(batch_size, self.window_size, self.c_out, device=self.device)
        sigmas = torch.zeros(batch_size, self.window_size, self.c_out, device=self.device)
        prev_y = x_enc[:, 0, :]  # Initialize with first value

        for t in range(self.window_size):
            # Update previous target value
            # Update previous target value based on position
            if t > 0:
                if t < self.seq_len:
                    # During the encoding phase, use actual values
                    prev_y = x_enc[:, t-1, :]
                else:
                    # During the prediction phase, use previous predictions
                    prev_y = means[:, t-1, :]

            # Get time features and covariates for current step
            time_feat = given_enc_embed[:, t, :]
            time_cov = x_mark_enc[:, t, :]

            # Ensure all tensors have the same batch dimension
            assert prev_y.shape[0] == batch_size, f"prev_y shape: {prev_y.shape}"
            assert time_feat.shape[0] == batch_size, f"time_feat shape: {time_feat.shape}"
            assert time_cov.shape[0] == batch_size, f"time_cov shape: {time_cov.shape}"
            assert meta_x_embed.shape[0] == batch_size, f"meta_x_embed shape: {meta_x_embed.shape}"
            
            # Concatenate all features
            inp_t = torch.cat([
                prev_y,
                time_feat,
                time_cov,
                meta_x_embed
            ], dim=-1)
            
            # Add sequence dimension
            inp_t = inp_t.unsqueeze(1)
            
            # LSTM step
            out, hidden = self.lstm(inp_t, hidden)
            
            # Project to target dimension
            mean_t = self.mean_layer(out.squeeze(1))
            log_sigma_t = self.sigma_layer(out.squeeze(1))

            means[:, t, :] = mean_t
            sigmas[:, t, :] = F.softplus(log_sigma_t)

            if not self.is_train:
                # During inference, sample from the distribution
                distribution = torch.distributions.Normal(mean_t, sigmas[:, t, :])
                means[:, t, :] = distribution.sample()
        
        # Denormalize if needed
        if self.use_norm:
            means = means * norm_std + norm_mean

        if output_attention:
            return means, None
        return means

    def _select_criterion(self):
      def gaussian_likelihood_loss(mu, sigma, labels):
        # Using the negative log likelihood of a Gaussian
        sigma = F.softplus(sigma)  # Ensure positivity
        distribution = torch.distributions.Normal(mu, sigma)
        return -distribution.log_prob(labels).mean()

      self.criterion = gaussian_likelihood_loss

    def set_train_mode(self):
        self.is_train = True

    def set_eval_mode(self):
        self.is_train = False

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        Shape:
        given_enc: (batch_size, window_size, time_num + time_cat)
        x_enc: (batch_size, seq_len, tgt_num)
        x_mark_enc: (batch_size, window_size, time_cov_size)
        meta_x: (batch_size, meta_num + meta_cat)
        """

        assert not (not self.output_attention and output_attention), \
            'model is not configured to output attention'

        # Add dimension checks
        assert given_enc.size(1) == self.window_size, \
            f"given_enc sequence length ({given_enc.size(1)}) must match window_size ({self.window_size})"
        assert x_enc.size(1) == self.seq_len, \
            f"x_enc sequence length ({x_enc.size(1)}) must match seq_len ({self.seq_len})"
        assert x_mark_enc.size(1) == self.window_size, \
            f"x_mark_enc sequence length ({x_mark_enc.size(1)}) must match window_size ({self.window_size})"

        if self.task_name == 'forecast':
            if output_attention:
                dec_out, attns = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention)
                return dec_out[:, -self.pred_len:, :], attns
            else:
                dec_out = self.forecast(given_enc, x_enc, x_mark_enc, meta_x, output_attention)
                return dec_out[:, -self.pred_len:, :]
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')