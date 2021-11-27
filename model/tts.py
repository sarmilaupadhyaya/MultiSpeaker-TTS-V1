# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from model.expressivity_encoder import EmbeddingNetwork


class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_enc_channels, filter_channels, filter_channels_dp,
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                 n_feats, dec_dim, beta_min, beta_max,
                 pe_scale, n_speakers, n_langs, gin_channels1, gin_channels2,
                 speaker_representation, language_representation):
                 # n_emotions -> n_langs
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.n_speakers = n_speakers
        self.gin_channels1 = gin_channels1
        self.n_langs = n_langs #
        self.gin_channels2 = gin_channels2

        self.speaker_representation = speaker_representation
        self.language_representation = language_representation

        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels,
                                   filter_channels, filter_channels_dp, n_heads,
                                   n_enc_layers, enc_kernel, enc_dropout,
                                   window_size, gin_channels1, gin_channels2)
        self.decoder = Diffusion(n_feats, dec_dim, beta_min, beta_max, pe_scale,
                                 gin_channels1, gin_channels2)

        ########################
        # depending on the version of the model, we choose the correct
        # representation for speaker and language.
        # it's either a simple Embedding or an Embedding network
        if n_speakers > 1:
            if speaker_representation == 'id':
                self.emb_g1 = nn.Embedding(n_speakers, gin_channels1)
                nn.init.uniform_(self.emb_g1.weight, -0.1, 0.1)
            else:
                self.emb_g1 = EmbeddingNetwork(n_feats, gin_channels1)

        # if both speaker and language representations are 'embedding',
        # then we use the same n_feats for them: is this correct?

        if n_langs > 1:
            if language_representation == 'id':
                self.emb_g2 = nn.Embedding(n_langs, gin_channels2)
                nn.init.uniform_(self.emb_g2.weight, -0.1, 0.1)
            else:
                self.emb_g2 = EmbeddingNetwork(n_feats, gin_channels2) # 10 X 2012 X 80
        #########################




    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, g1=None, g2=None, temperature=1.0, stoc=False, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """



        #print(g.shape, 'g dim')
        x, x_lengths, g1, g2 = self.relocate_input([x, x_lengths, g1, g2])
        #print(g.shape, g)
        if g1 is not None:
            #print(g.shape, 'before nn.embedding')
            g1 = F.normalize(self.emb_g1(g1)).permute(0,2,1)#.unsqueeze(-1)

        if g2 is not None:
            #print(g.shape, 'before nn.embedding')
            g2 = F.normalize(self.emb_g2(g2.permute(0,2,1)))

        #print(g1.shape, g2.shape, 'after embedding extraction ')# Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, g1=g1, g2=g2)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, g1, g2, stoc)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, g1=None, g2=None,out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """

        #print(g.shape, 'g dim')
        x, x_lengths, y, y_lengths, g1, g2 = self.relocate_input([x, x_lengths, y, y_lengths, g1, g2])

        #print(g2.shape, 'before embedding')

        if g1 is not None:
            g1 = F.normalize(self.emb_g1(g1)).permute(0,2,1)#.unsqueeze(-1)

        if g2 is not None:
            #print(g2.shape, 'mel shape as input to exp encoder')
            g2 = F.normalize(self.emb_g2(g2.permute(0,2,1)))#.unsqueeze(-1)
            #print(g2.shape, 'output shape of encoding 16, 1, 80')
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, g1=g1, g2=g2)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        # Compute loss of score-based decoder
        #print('before diffusion', y.shape, y_mask.shape, mu_y.shape, g.shape)
        diff_loss = self.decoder.compute_loss(y, y_mask, mu_y, g1, g2)

        return dur_loss, prior_loss, diff_loss
