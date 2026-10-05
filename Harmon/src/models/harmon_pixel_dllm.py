"""Pixel-only masked denoising; all other tasks inherit HarmonDev unchanged."""
import math

import torch
from torch import nn
from tqdm import trange
from xtuner.utils import IMAGE_TOKEN_INDEX

from .harmon import mask_by_order
from .harmon_dev import HarmonDev, _ScaleGradient


class HarmonPixelDLLM(HarmonDev):
    def __init__(self, pixel_dllm=True, **kwargs):
        super().__init__(**kwargs)
        self.pixel_dllm = pixel_dllm
        hidden = self.llm.config.hidden_size
        self.pixel_mask_token = nn.Parameter(torch.empty(1, 1, hidden))
        self.pixel_query_proj = nn.Linear(hidden, self.mar.decoder_embed_dim)
        nn.init.normal_(self.pixel_mask_token, std=0.02)
        nn.init.zeros_(self.pixel_query_proj.weight)
        nn.init.zeros_(self.pixel_query_proj.bias)

    def _pixel_condition(self, latents, input_ids, attention_mask):
        """Replace the single source-image placeholder with its visual features."""
        if not torch.all(input_ids[:, 3] == IMAGE_TOKEN_INDEX):
            raise ValueError('Expected the source-image placeholder at position 3')
        _, visual = self.extract_visual_feature(latents)
        if self.grad_scale is not None:
            visual = _ScaleGradient.apply(visual, self.grad_scale)
        embed = self.llm.get_input_embeddings()
        condition = torch.cat((embed(input_ids[:, :3]), visual,
                               embed(input_ids[:, 4:])), dim=1)
        valid = torch.cat((attention_mask[:, :3].bool(),
                           attention_mask.new_ones(visual.shape[:2]).bool(),
                           attention_mask[:, 4:].bool()), dim=1)
        return condition, valid

    @staticmethod
    def _pixel_attention(valid, target_len, dtype):
        """Causal prefix, bidirectional target, no target-to-prefix information flow."""
        batch, prefix_len = valid.shape
        length = prefix_len + target_len
        allowed = torch.ones(length, length, dtype=torch.bool, device=valid.device)
        allowed[:prefix_len, :prefix_len] = torch.ones(
            prefix_len, prefix_len, dtype=torch.bool, device=valid.device).tril()
        allowed[:prefix_len, prefix_len:] = False
        key_valid = torch.cat((valid, valid.new_ones(batch, target_len)), dim=1)
        allowed = allowed[None, None] & key_valid[:, None, None, :]
        attention = torch.zeros(allowed.shape, dtype=dtype, device=valid.device)
        attention.masked_fill_(~allowed, torch.finfo(dtype).min)
        positions = key_valid.long().cumsum(-1).sub(1).clamp_min(0)
        return attention, positions

    def forward_pixel_dllm(self, latents, mask, condition, valid):
        batch, height, width, _ = latents.shape
        buffer = self.mar.buffer_size
        # Masked target values are discarded inside MAR before any attention.
        encoded, visible_embeds = self.extract_visual_feature(latents, mask=mask)
        visible = torch.cat((~mask.bool(), mask.new_ones(batch, buffer).bool()), dim=1)
        spatial_pos = self.mar.get_encoder_pos_embed(height, width)[:, buffer:]
        queries = self.pixel_mask_token + self.proj_in(spatial_pos)
        target = torch.cat((queries.expand(batch, -1, -1),
                            visible_embeds[:, -buffer:]), dim=1).clone()
        target[visible] = visible_embeds.reshape(-1, visible_embeds.shape[-1])
        inputs = torch.cat((condition, target), dim=1)
        attention, positions = self._pixel_attention(valid, target.shape[1], inputs.dtype)
        hidden = self.llm_model(inputs_embeds=inputs, attention_mask=attention,
                                position_ids=positions, use_cache=False,
                                return_dict=True).last_hidden_state[:, -target.shape[1]:]
        # Qwen uses [spatial, buffer]; MAR uses [buffer, spatial].
        visible_hidden = hidden[visible].reshape(batch, -1, hidden.shape[-1])
        visible_hidden = torch.cat((visible_hidden[:, -buffer:],
                                    visible_hidden[:, :-buffer]), dim=1)
        encoded = encoded + self.proj_out(visible_hidden)
        residual = self.pixel_query_proj(hidden)
        residual = torch.cat((residual[:, -buffer:], residual[:, :-buffer]), dim=1)
        return self.mar.forward_mae_decoder(
            encoded, mask, image_shape=(height, width), masked_query_residual=residual)

    def recon_loss(self, data_dict):
        if not self.pixel_dllm:
            return super().recon_loss(data_dict)
        latents = self.encode(data_dict['pixel_values'].to(device=self.device, dtype=self.dtype))
        condition, valid = self._pixel_condition(
            latents, data_dict['input_ids'].to(self.device),
            data_dict['attention_mask'].to(self.device))
        target = latents.flatten(1, 2)
        orders = self.mar.sample_orders(target.shape[0], seq_len=target.shape[1])
        mask = self.mar.random_masking(target, orders)
        decoded = self.forward_pixel_dllm(latents, mask, condition, valid)
        return self.mar.forward_loss(z=decoded, target=target.detach(), mask=mask)

    @torch.no_grad()
    def sample_pixel_dllm(self, image, prompt='Describe the image in detail.',
                          num_iter=32, temperature=1.0, cfg=1.0, progress=True):
        if cfg != 1.0:
            raise ValueError('Stage 1 uses CFG=1; image-condition dropout is not trained')
        if num_iter < 1:
            raise ValueError('num_iter must be positive')
        source = self.encode(image.to(device=self.device, dtype=self.dtype))
        batch, height, width, channels = source.shape
        text = self.prompt_template['INSTRUCTION'].format(input=prompt)
        ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(self.device)
        ids = torch.cat((ids[:, :3], ids.new_full((1, 1), IMAGE_TOKEN_INDEX), ids[:, 3:]), dim=1)
        ids = ids.expand(batch, -1)
        condition, valid = self._pixel_condition(source, ids, torch.ones_like(ids, dtype=torch.bool))
        length = height * width
        tokens = source.new_zeros(batch, length, channels)
        mask = torch.ones(batch, length, dtype=torch.bool, device=self.device)
        orders = self.mar.sample_orders(batch, seq_len=length)
        for step in trange(num_iter, disable=not progress):
            decoded = self.forward_pixel_dllm(
                tokens.reshape(batch, height, width, channels), mask.to(self.dtype), condition, valid)
            remaining = int(mask[0].sum().item())
            next_count = 0 if step == num_iter - 1 else max(
                0, min(remaining - 1, math.floor(length * math.cos(math.pi / 2 * (step + 1) / num_iter))))
            next_mask = mask_by_order(torch.tensor(next_count, device=self.device), orders, batch, length)
            commit = mask & ~next_mask
            if commit.any():
                tokens[commit] = self.mar.diffloss.sample(
                    decoded[commit], temperature=temperature, cfg=1.0).to(tokens.dtype)
            mask = next_mask
            if not mask.any():
                break
        return self.decode(tokens.reshape(batch, height, width, channels))
