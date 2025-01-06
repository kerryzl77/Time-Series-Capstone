import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class DeepAR_AS(nn.Module):
    def __init__(self, configs, device):
        super(DeepAR_AS, self).__init__()
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

        # Calculate input size
        time_feat_size = self.time_num + sum(configs.time_cat_embed)
        meta_feat_size = self.meta_num + sum(configs.meta_cat_embed)
        self.input_size = (
            self.tgt_num  # previous target values
            + time_feat_size  # time features
            + self.time_cov_size  # time covariates
            + meta_feat_size  # meta features
        )

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
            meta_x_embed = torch.zeros(b_sz, self.meta_num + sum(self.meta_cat_embed)).to(self.device)
            
        return given_enc_embed, meta_x_embed

    def _get_distribution_params(self, given_enc, x_enc, x_mark_enc, meta_x):
        """Get distribution parameters (mean, sigma) for the entire sequence"""
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

        means = torch.zeros(batch_size, self.window_size, self.c_out, device=self.device)
        sigmas = torch.zeros(batch_size, self.window_size, self.c_out, device=self.device)
        prev_y = x_enc[:, 0, :]

        for t in range(self.window_size):
            # Update previous target value based on training/inference mode
            if t > 0:
                if self.is_train or t < self.seq_len:
                    prev_y = x_enc[:, min(t-1, self.seq_len-1), :]
                else:
                    prev_y = means[:, t-1, :]

            # Get time features and covariates
            time_feat = given_enc_embed[:, t, :]
            time_cov = x_mark_enc[:, t, :]
            
            # Concatenate inputs
            inp_t = torch.cat([prev_y, time_feat, time_cov, meta_x_embed], dim=-1)
            inp_t = inp_t.unsqueeze(1)
            
            # LSTM step
            out, hidden = self.lstm(inp_t, hidden)
            
            # Get distribution parameters
            means[:, t, :] = self.mean_layer(out.squeeze(1))
            sigmas[:, t, :] = F.softplus(self.sigma_layer(out.squeeze(1)))

        if self.use_norm:
            means = means * norm_std + norm_mean
            sigmas = sigmas * norm_std

        return means, sigmas

    def gaussian_nll_loss(self, pred, target):
      """Compute negative log-likelihood loss"""
      # print("Using Gaussian NLL Loss")
      # Extract means and sigmas from the concatenated pred tensor
      c_out = self.c_out
      means = pred[..., :c_out]  # First half of channels
      sigmas = pred[..., c_out:]  # Second half of channels
      
      # Create distribution without additional slicing
      distribution = torch.distributions.Normal(means, sigmas)
      return -distribution.log_prob(target).sum(dim=(1, 2)).mean()

    def forward(self, given_enc, x_enc, x_mark_enc, meta_x=None, output_attention=False):
        """
        During training: Return both means and sigmas packaged together
        During inference: Return samples from the distribution
        """
        means, sigmas = self._get_distribution_params(given_enc, x_enc, x_mark_enc, meta_x)
        
        if self.task_name == 'forecast':
            if self.is_train:
                # Store sigmas as model attribute for loss computation
                self.current_sigmas = sigmas
                # Return both means and sigmas together as a single tensor
                # Shape: [batch_size, seq_len, 2 * c_out]
                return torch.cat([means, sigmas], dim=-1)
            else:
                # During inference, sample from the distribution
                distribution = torch.distributions.Normal(
                    means[:, -self.pred_len:, :],
                    sigmas[:, -self.pred_len:, :]
                )
                samples = distribution.sample()
                if output_attention:
                    return samples, None
                return samples
        else:
            raise ValueError(f'Unknown task_name: {self.task_name}')

    def set_train_mode(self):
        """Set model to training mode"""
        self.is_train = True
        self.train()

    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self.is_train = False
        self.eval()