import torch
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX
from transformers.cache_utils import DynamicCache
from .harmon import Harmon
from torch.nn.utils.rnn import pad_sequence

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class HarmonDev(Harmon, BaseModel):
    def __init__(self,
                 grad_scale=0.1,
                 loss_weights={'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0, 'edit': 1.0},
                 pretrained_pth=None,
                 freeze_llm=False,
                 freeze_mar_encoder=False,
                 freeze_mar_decoder=False,
                 freeze_proj_in=False,
                 freeze_proj_out=False,
                 gradient_checkpointing=True,
                 dllm=False,
                 block_size=32,
                 prior_dist = 'Mask',
                 min_mask_rate = 0.001,
                 max_mask_rate = 1.0,
                 lora=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dllm = dllm
        self.min_mask_rate = min_mask_rate
        self.max_mask_rate = max_mask_rate

        self.block_size = block_size
        self.prior_dist = prior_dist
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.debuged = False
        self._input_require_grads_enabled = False

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        # Freeze specific modules
        if freeze_llm:
            self.llm.requires_grad_(False)
            print_log('Frozen LLM')
        
        if freeze_mar_encoder:
            # Freeze MAR encoder blocks and related components
            for param in self.mar.encoder_blocks.parameters():
                param.requires_grad = False
            self.mar.z_proj.requires_grad_(False)
            self.mar.z_proj_ln.requires_grad_(False)
            self.mar.encoder_pos_embed_learned.requires_grad = False
            self.mar.encoder_norm.requires_grad_(False)
            self.mar.class_emb.requires_grad_(False)
            self.mar.fake_latent.requires_grad = False
            print_log('Frozen MAR Encoder')
        
        if freeze_mar_decoder:
            # Freeze MAR decoder blocks and related components
            for param in self.mar.decoder_blocks.parameters():
                param.requires_grad = False
            self.mar.decoder_embed.requires_grad_(False)
            self.mar.mask_token.requires_grad = False
            self.mar.decoder_pos_embed_learned.requires_grad = False
            self.mar.decoder_norm.requires_grad_(False)
            self.mar.diffusion_pos_embed_learned.requires_grad = False
            self.mar.diffloss.requires_grad_(False)
            print_log('Frozen MAR Decoder')
        
        if freeze_proj_in:
            self.proj_in.requires_grad_(False)
            print_log('Frozen proj_in')
        
        if freeze_proj_out:
            self.proj_out.requires_grad_(False)
            print_log('Frozen proj_out')

        if lora is not None:
            self._setup_lora(lora)

        # gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def _setup_lora(self, lora):
        if not isinstance(lora, dict):
            raise TypeError(
                '`lora` must be a dict containing PEFT LoraConfig fields, '
                f'but got {type(lora).__name__}.')

        lora_config_dict = dict(lora)
        target_modules = lora_config_dict.get('target_modules')
        if not target_modules:
            raise ValueError(
                '`lora.target_modules` must explicitly select the Qwen '
                'modules that receive LoRA adapters.')

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise ImportError(
                'LoRA training requires PEFT. Install it in the training '
                'environment before setting `model.lora`.') from error

        lora_config = LoraConfig(**lora_config_dict)
        self.llm = get_peft_model(self.llm, lora_config)

        trainable_params = sum(
            parameter.numel()
            for parameter in self.llm.parameters()
            if parameter.requires_grad)
        total_params = sum(
            parameter.numel() for parameter in self.llm.parameters())
        print_log(
            f'Enabled LoRA: {trainable_params:,} trainable LLM parameters '
            f'out of {total_params:,}.')

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if self._input_require_grads_enabled:
            self.llm.disable_input_require_grads()
            self._input_require_grads_enabled = False
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if (hasattr(self.llm, 'peft_config')
                and not self._input_require_grads_enabled):
            self.llm.enable_input_require_grads()
            self._input_require_grads_enabled = True
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k}

        return state_dict

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        x_enc = self.forward_mae_encoder(x, mask, input_ids=input_ids,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def image2text_loss(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        # 获取 dLLM 特有的 masks (如果存在)
        response_mask = data_dict.get('response_mask', None)
        if response_mask is not None:
            response_mask = response_mask.to(self.device)
            
        loss_mask = data_dict.get('loss_mask', None)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        # 【新增】：取出 create_dllm_batch 构造的 position_ids（仅 dllm 分支存在）
        position_ids = data_dict.get('position_ids', None)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            assert False
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            for i in range(input_ids.shape[0]):
                assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
            input_ids = torch.cat(
                (
                    input_ids[:, :3],
                    IMAGE_TOKEN_INDEX * torch.ones(
                        input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                    input_ids[:, 4:],
                ), dim=1
            )
            if attention_mask.dim() == 4:
                # 4D Mask: [bsz, 1, q_len, k_len]
                # 1. 沿 Query 维度 (dim=2) 复制图像占位符的可见性
                attention_mask = torch.cat([
                    attention_mask[:, :, :3, :],
                    attention_mask[:, :, 3:4, :].repeat(1, 1, 1088, 1),
                    attention_mask[:, :, 4:, :]
                ], dim=2)
                
                # 2. 沿 Key 维度 (dim=3) 复制图像占位符的可见性
                attention_mask = torch.cat([
                    attention_mask[:, :, :, :3],
                    attention_mask[:, :, :, 3:4].repeat(1, 1, 1, 1088),
                    attention_mask[:, :, :, 4:]
                ], dim=3)
            else:
                # 兼容传统的 2D Mask: [bsz, seq_len]
                attention_mask = torch.cat(
                    (
                        attention_mask[:, :3],
                        torch.ones(
                            attention_mask.shape[0], 1088, dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[:, 4:],
                    ), dim=1
                )
            
            labels = torch.cat(
                (
                    labels[:, :3],
                    labels[0, 3] * torch.ones(
                        labels.shape[0], 1088 , dtype=torch.long, device=labels.device),
                    labels[:, 4:],
                ), dim=1
            )
            if response_mask is not None:
                response_mask = torch.cat(
                    (
                        response_mask[:, :3],
                        torch.zeros(response_mask.shape[0], 1088, dtype=torch.bool, device=response_mask.device),
                        response_mask[:, 4:],
                    ), dim=1
                )
                # 存回字典，供后续提取 Logits 使用
                data_dict['response_mask'] = response_mask
                
            if loss_mask is not None:
                loss_mask = torch.cat(
                    (
                        loss_mask[:, :3],
                        torch.zeros(loss_mask.shape[0], 1088, dtype=torch.bool, device=loss_mask.device),
                        loss_mask[:, 4:],
                    ), dim=1
                )
                data_dict['loss_mask'] = loss_mask

            # 【新增】：图像占位 1→1088 同步扩展 position_ids，保持与 input_ids 对齐，
            # 且重新映射到“im_start user...IMG_1...IMG_1088...”的康康位置，保证 RoPE 与纯 AR 一致。
            if position_ids is not None:
                base_pos_at_img = position_ids[:, 3:4]  # [b, 1]
                img_offsets = torch.arange(1088, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(0)
                image_pos_ids = base_pos_at_img + img_offsets  # [b, 1088]
                after_pos = position_ids[:, 4:] + 1087
                position_ids = torch.cat([position_ids[:, :3], image_pos_ids, after_pos], dim=1)

            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            loss_null = 0.0

        if getattr(self, "dllm", False) and attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=inputs_embeds.dtype)
            float_mask = float_mask.masked_fill(~attention_mask, torch.finfo(inputs_embeds.dtype).min)
            attention_mask = float_mask

        # 【修改】：dllm 分支显式传 position_ids，否则 HF 会自动 arange 导致 clean/noisy 段数据分布不对齐。
        llm_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if getattr(self, "dllm", False) and position_ids is not None:
            llm_kwargs['position_ids'] = position_ids

        output = self.llm_model(**llm_kwargs)

        if getattr(self, "dllm", False):
            logits2keep = data_dict['loss_mask'] & data_dict['response_mask']
            last_hidden_state = output.last_hidden_state#[:, :-1]
            # labels = labels[:, 1:]
            last_hidden_state = last_hidden_state[logits2keep]
            labels = labels[logits2keep]
            logits = self.llm.get_output_embeddings()(last_hidden_state)

            loss_i2t = F.cross_entropy(input=logits, target=labels)
        else:
            last_hidden_state = output.last_hidden_state[:, :-1]
            labels = labels[:, 1:]
            last_hidden_state = last_hidden_state[labels >= 0]
            labels = labels[labels >= 0]
            logits = self.llm.get_output_embeddings()(last_hidden_state)

            loss_i2t = F.cross_entropy(input=logits, target=labels)

        return loss_i2t + loss_null

    def recon_loss(self, data_dict):

        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        _, z_enc = self.extract_visual_feature(x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def edit_loss(self, data_dict):
        """Image editing loss.
        
        Unlike reconstruction which uses the same image as input/output,
        editing uses source_image as condition and edited_image as target.
        """
        # Encode source image (condition)
        source_x = data_dict['source_pixel_values'].to(dtype=self.dtype, device=self.device)
        source_x = self.encode(source_x)  # b m n c
        
        # Encode target edited image
        target_x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        target_x = self.encode(target_x)   # b m n c
        b, m, n, _ = target_x.shape
        gt_latents = target_x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        # Extract visual features from source image (as condition)
        _, z_enc = self.extract_visual_feature(source_x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        # Generate target edited image
        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(target_x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            target_x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def joint_loss(self, data_dict):
        """Joint loss for SFT and Depth estimation in a single step.
        
        This method manually batches inputs for SFT and Depth tasks to avoid
        calling shared modules (like Visual Encoder and LLM) multiple times
        in a single forward pass, which causes "Gradient computed twice" errors in DeepSpeed.
        """
        # --- 1. Prepare Inputs ---
        
        # SFT Data
        sft_input_ids = data_dict['sft_input_ids'].to(self.device)
        sft_labels = data_dict['sft_labels'].to(self.device)
        sft_attention_mask = data_dict['sft_attention_mask'].to(self.device)
        sft_pixel_values = data_dict['sft_pixel_values'].to(dtype=self.dtype, device=self.device)
        
        # Depth Data
        depth_source_pixel_values = data_dict['depth_source_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_target_pixel_values = data_dict['depth_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_input_ids = data_dict['depth_input_ids'].to(self.device)
        depth_attention_mask = data_dict['depth_attention_mask'].to(self.device)
        
        b_sft = sft_input_ids.shape[0]
        b_depth = depth_input_ids.shape[0]
        
        # --- 2. Shared VAE Encoding (Safe to call multiple times if frozen, but efficient to batch) ---
        # Concatenate all images for encoding: [SFT, Depth_Source, Depth_Target]
        all_images = torch.cat([sft_pixel_values, depth_source_pixel_values, depth_target_pixel_values], dim=0)
        all_latents = self.encode(all_images)
        
        sft_latents, depth_source_latents, depth_target_latents = torch.split(all_latents, [b_sft, b_depth, b_depth], dim=0)
        
        # --- 3. Shared Visual Encoder (MUST batch this if trainable) ---
        # Concatenate latents for visual feature extraction: [SFT, Depth_Source]
        # Note: Depth_Target also needs visual features inside forward_mae_encoder, but let's see.
        # In forward_mae_encoder(target_x), it calls extract_visual_feature(target_x).
        # So we need to extract features for ALL 3 sets of latents.
        
        all_vis_latents = torch.cat([sft_latents, depth_source_latents, depth_target_latents], dim=0)
        
        # We also need to generate masks for Depth_Target (for MAR)
        # SFT and Depth_Source don't use masks (mask=None/zeros) for feature extraction usually
        # But forward_mae_encoder passes a mask.
        
        # Generate masks for Depth_Target
        orders = self.mar.sample_orders(bsz=b_depth, seq_len=depth_target_latents.shape[1]*depth_target_latents.shape[2])
        depth_mask = self.mar.random_masking(depth_target_latents.flatten(1, 2), orders)
        
        # For SFT and Source, mask is zeros (no masking)
        # We need to create a combined mask for batching
        # Mask shape for one image: [b, mn]
        mask_zeros = torch.zeros(b_sft + b_depth, depth_mask.shape[1], device=self.device, dtype=depth_mask.dtype)
        all_masks = torch.cat([mask_zeros, depth_mask], dim=0)
        
        # Extract Visual Features (One pass)
        # This calls mar.forward_mae_encoder -> BUT wait, forward_mae_encoder calls llm_model!
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder. 
        # Is self.mar.forward_mae_encoder the same as self.forward_mae_encoder?
        # NO. self.mar is the MAR module; self.forward_mae_encoder is a method on the Harmon class.
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder.
        # Harmon.forward_mae_encoder calls extract_visual_feature AND llm_model.
        
        # So extract_visual_feature runs the Vision Transformer.
        # We run this ONCE.
        
        all_x_enc, all_z_enc = self.extract_visual_feature(all_vis_latents, mask=all_masks)
        
        # Split results
        # z_enc is the projected features used as inputs to LLM
        # x_enc is the features from Vision Encoder (before projection?) No, extract_visual_feature returns proj_in(x_enc).
        # Actually: x_enc from mar.forward_mae_encoder, z_enc = proj_in(x_enc).
        # We need z_enc for SFT and Depth Source.
        # We need x_enc for Depth Target (residual connection).
        
        z_enc_sft, z_enc_source, z_enc_target = torch.split(all_z_enc, [b_sft, b_depth, b_depth], dim=0)
        x_enc_target = torch.split(all_x_enc, [b_sft, b_depth, b_depth], dim=0)[2]
        
        # Apply scale gradient if needed
        if self.grad_scale is not None:
            z_enc_sft = _ScaleGradient.apply(z_enc_sft, self.grad_scale)
            z_enc_source = _ScaleGradient.apply(z_enc_source, self.grad_scale)
            # Target features usually don't get scaled? edit_loss doesn't scale target features explicitly, 
            # but forward_mae_encoder calls extract_visual_feature which returns z_enc.
            # In forward_mae_encoder, z_enc is used as input to LLM.
            # Does it scale? In edit_loss code:
            # x_enc = self.forward_mae_encoder(...)
            # Inside forward_mae_encoder: x_enc, z_enc = self.extract_visual_feature(..., detach=detach)
            # It doesn't seem to apply scale gradient inside forward_mae_encoder.
            # So we apply it only for SFT and Source (Condition).
        
        # --- 4. Prepare LLM Inputs ---
        
        # Get actual visual token length from encoder output
        vis_token_len = z_enc_sft.shape[1]

        # A. SFT Inputs construction (Logic from image2text_loss)
        for i in range(sft_input_ids.shape[0]):
            assert sft_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        # Insert Image Tokens placeholders
        sft_input_ids_exp = torch.cat((
            sft_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_input_ids[:, 4:]), dim=1)
            
        sft_attention_mask_exp = torch.cat((
            sft_attention_mask[:, :3],
            torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_attention_mask[:, 4:]), dim=1)
            
        # SFT Labels (for loss later)
        sft_labels_exp = torch.cat((
            sft_labels[:, :3],
            sft_labels[0, 3] * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_labels[:, 4:]), dim=1)
            
        # Embeddings
        sft_inputs_embeds = z_enc_sft.new_zeros(*sft_input_ids_exp.shape, self.llm.config.hidden_size)
        sft_inputs_embeds[sft_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_sft.flatten(0, 1)
        mask_text = sft_input_ids_exp != IMAGE_TOKEN_INDEX
        sft_inputs_embeds[mask_text] = self.llm.get_input_embeddings()(sft_input_ids_exp[mask_text])
        
        # B. Depth Inputs construction (Logic from edit_loss -> forward_mae_encoder)
        # First prepare the "Condition" embeddings (Source Image + Prompt)
        for i in range(depth_input_ids.shape[0]):
            assert depth_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        depth_input_ids_exp = torch.cat((
            depth_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_input_ids[:, 4:]), dim=1)
            
        depth_attention_mask_exp = torch.cat((
            depth_attention_mask[:, :3],
            torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_attention_mask[:, 4:]), dim=1)
            
        # Condition Embeddings (Source + Prompt)
        depth_cond_embeds = z_enc_source.new_zeros(*depth_input_ids_exp.shape, self.llm.config.hidden_size)
        depth_cond_embeds[depth_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_source.flatten(0, 1)
        mask_cond_text = depth_input_ids_exp != IMAGE_TOKEN_INDEX
        depth_cond_embeds[mask_cond_text] = self.llm.get_input_embeddings()(depth_input_ids_exp[mask_cond_text])
        
        # Now combine Condition with Target (Logic from prepare_forward_input)
        # inputs_embeds = torch.cat([inputs_embeds, x], dim=1) where x is z_enc_target
        depth_full_embeds = torch.cat([depth_cond_embeds, z_enc_target], dim=1)
        
        # Prepare Mask for Depth (Logic from prepare_forward_input)
        # attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)
        l_target = z_enc_target.shape[1]
        depth_full_mask = torch.cat([
            depth_attention_mask_exp, 
            depth_attention_mask_exp.new_ones(b_depth, l_target)
        ], dim=1)
        
        # --- 5. Batch LLM Forward ---
        # We need to concat sft_inputs_embeds and depth_full_embeds.
        # They might have different lengths. We need to pad.
        
        len_sft = sft_inputs_embeds.shape[1]
        len_depth = depth_full_embeds.shape[1]
        max_len = max(len_sft, len_depth)
        
        # Pad SFT
        if len_sft < max_len:
            pad_len = max_len - len_sft
            pad_embeds = torch.zeros(b_sft, pad_len, sft_inputs_embeds.shape[2], device=self.device, dtype=self.dtype)
            sft_inputs_embeds = torch.cat([sft_inputs_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_sft, pad_len, device=self.device, dtype=sft_attention_mask_exp.dtype)
            sft_attention_mask_exp = torch.cat([sft_attention_mask_exp, pad_mask], dim=1)
            
        # Pad Depth
        if len_depth < max_len:
            pad_len = max_len - len_depth
            pad_embeds = torch.zeros(b_depth, pad_len, depth_full_embeds.shape[2], device=self.device, dtype=self.dtype)
            depth_full_embeds = torch.cat([depth_full_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_depth, pad_len, device=self.device, dtype=depth_full_mask.dtype)
            depth_full_mask = torch.cat([depth_full_mask, pad_mask], dim=1)
            
        # Concat
        joint_inputs_embeds = torch.cat([sft_inputs_embeds, depth_full_embeds], dim=0)
        joint_attention_mask = torch.cat([sft_attention_mask_exp, depth_full_mask], dim=0)
        
        # Forward LLM (ONE PASS!)
        output = self.llm_model(inputs_embeds=joint_inputs_embeds,
                                attention_mask=joint_attention_mask,
                                return_dict=True)
        
        last_hidden_state = output.last_hidden_state
        
        # --- 6. Compute Losses ---
        
        # A. SFT Loss
        # Slice out SFT part
        sft_hidden = last_hidden_state[:b_sft, :len_sft] # Remove padding
        
        # Logic from image2text_loss
        # shift labels and hidden states
        sft_hidden = sft_hidden[:, :-1]
        sft_labels_shifted = sft_labels_exp[:, 1:]
        
        # Flatten and filter
        # Note: sft_labels_exp was not padded in Step 5, so we use original shape logic
        # BUT sft_hidden is taken from padded batch. We sliced to len_sft.
        # However, sft_labels_exp matches len_sft (the length before padding).
        
        sft_hidden_flat = sft_hidden[sft_labels_shifted >= 0]
        sft_labels_flat = sft_labels_shifted[sft_labels_shifted >= 0]
        
        sft_logits = self.llm.get_output_embeddings()(sft_hidden_flat)
        loss_sft = F.cross_entropy(input=sft_logits, target=sft_labels_flat)
        
        # B. Depth Loss
        # Slice out Depth part
        depth_hidden = last_hidden_state[b_sft:, :len_depth]
        
        # Logic from forward_mae_encoder: z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]
        # z_enc here refers to z_enc_target (length 256 usually)
        # depth_hidden contains [Condition, Target]. We want the part corresponding to Target.
        # Target length is l_target
        z_llm_depth = depth_hidden[:, -l_target:]
        
        # Logic: move buffers back (from forward_mae_encoder)
        z_llm_depth = torch.cat([
            z_llm_depth[:, -self.mar.buffer_size:],
            z_llm_depth[:, :-self.mar.buffer_size]], dim=1)
            
        # Residual learning
        x_enc_final = x_enc_target + self.proj_out(z_llm_depth)
        
        # Decoder and Loss
        gt_latents = depth_target_latents.clone().detach().view(b_depth, -1, depth_target_latents.shape[-1]) # Flatten m*n
        z_dec = self.mar.forward_mae_decoder(x_enc_final, depth_mask, image_shape=(16, 16)) # Assuming 16x16 patches?
        # Need to get m, n from latents shape.
        # all_latents shape: [b, m, n, c]
        m, n = depth_target_latents.shape[1], depth_target_latents.shape[2]
        
        loss_depth = self.mar.forward_loss(z=z_dec, target=gt_latents, mask=depth_mask)
        
        return loss_sft, loss_depth

    def find_response_span(self, input_ids):
        n = len(input_ids)
        IM_END_ID = 151645
        header_prefix = [151644, 77091]
        m = len(header_prefix)

        start_idx = -1

        for i in range(n - m, -1, -1):
            # 匹配 <|im_start|>assistant\n
            if input_ids[i : i+m].tolist() == header_prefix:
                # 找到了 <|im_start|>assistant
                # 我们跳过这个 header 以及紧随其后的“换行/格式”Token
                # 无论它是 \n, \n\n 还是 .\n
                start_idx = i + m + 1 # 内容从 header 之后开始
                break

        if start_idx == -1: 
            return None, None

        end_idx = -1
        for i in range(start_idx, n):
            if input_ids[i].item() == IM_END_ID:
                end_idx = i + 1 # 包含这个 <|im_end|> 本身
                break

        # 没找到，取到序列末尾进行兜底。
        if end_idx == -1:
            end_idx = n 

        return start_idx, end_idx

    def _sample_block_times(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        t = torch.rand(num_blocks, device=device).clamp(low, high)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=t.dtype) / num_blocks
            t = t / num_blocks + offset
            t = t[torch.randperm(num_blocks, device=device)]
        return t
    
    def create_attention_mask(self, prefix_len, res_len, block_size, max_len):
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        # Prefix: 每个 token 独占一个 block_id，让 base_mask 退化为严格 token 级 causal，
        # 避免把 Qwen 预训练的因果分布破坏成块内双向。
        block_ids[:clean_start] = torch.arange(clean_start)
        token_types[:clean_start] = 0

        # 响应段的 block_id 从 prefix_len 起递增（保持与 prefix 完全隔离）
        response_block_offset = clean_start

        if clean_end > clean_start:
            clean_len = clean_end - clean_start
            clean_rel_positions = torch.arange(clean_len)
            clean_blocks = response_block_offset + clean_rel_positions // block_size
            block_ids[clean_start:clean_end] = clean_blocks
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            noisy_len = seq_len - noisy_start
            noisy_rel_positions = torch.arange(noisy_len)
            noisy_blocks = response_block_offset + noisy_rel_positions // block_size
            block_ids[noisy_start:seq_len] = noisy_blocks
            token_types[noisy_start:seq_len] = 2

        q_blocks = block_ids.view(-1, 1)
        k_blocks = block_ids.view(1, -1)
        q_types = token_types.view(-1, 1)
        k_types = token_types.view(1, -1)

        # base_mask：prefix/clean 的 query 走块级因果；同时禁止任何非 noisy query 看到 noisy K。
        # 这样 clean 段永远看不到 noisy 段，避免表征污染。
        base_mask = (q_blocks >= k_blocks) & (k_types != 2)
        # noisy query 的可见范围：全部 prefix + 严格前置 clean block + 同 block 的 noisy token。
        noisy_query_mask = q_types == 2
        noisy_visibility = (k_types == 0) | ((k_types == 1) & (k_blocks < q_blocks)) | ((k_types == 2) & (k_blocks == q_blocks))
        local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_dllm_batch(self, data_dict, mask_token_id=151671, pad_token_id=151645):
        block_size = getattr(self, "block_size", 32)

        for key in data_dict.keys():
            if 'input_ids' not in data_dict[key] or 'labels' not in data_dict[key]:
                continue
            
            data_batch = data_dict[key]
            input_ids_list = data_batch['input_ids']
            labels_list = data_batch['labels']
            bsz = len(input_ids_list)

            batch_input_ids, batch_labels = [], []
            batch_t_types, batch_b_indices = [], []
            batch_response_mask, batch_loss_mask = [], [] # 【新增】：容器
            batch_position_ids = []  # 【新增】：per-sample position_ids，用于对齐 clean/noisy
            batch_lengths = []
            batch_t = []

            for i in range(bsz):
                i_ids = input_ids_list[i].clone()
                l_ids = labels_list[i].clone()
                
                try:
                    start_idx, end_idx = self.find_response_span(i_ids)
                except ValueError:
                    start_idx, end_idx = 0, 0
                    
                len_res = end_idx - start_idx
                
                if len_res > 0:
                    ar_i_ids = i_ids[:end_idx]
                    ar_l_ids = l_ids[:end_idx]
                    
                    # 补齐到 block_size 的倍数
                    remainder = len_res % block_size
                    pad_len = block_size - remainder if remainder != 0 else 0
                    
                    if pad_len > 0:
                        pad_tensor = torch.full((pad_len,), pad_token_id, dtype=ar_i_ids.dtype, device=ar_i_ids.device)
                        lbl_pad_tensor = torch.full((pad_len,), -100, dtype=ar_l_ids.dtype, device=ar_l_ids.device)
                        ar_i_ids = torch.cat([ar_i_ids, pad_tensor])
                        ar_l_ids = torch.cat([ar_l_ids, lbl_pad_tensor])
                        end_idx += pad_len
                        len_res += pad_len
                    
                    # AR 区域属性
                    ar_t_types = torch.zeros_like(ar_i_ids)
                    ar_b_indices = torch.zeros_like(ar_i_ids)
                    ar_t_types[start_idx:end_idx] = 1
                    
                    # 【新增】：AR 区域的 Mask 初始化 (全为 False)
                    ar_response_mask = torch.zeros_like(ar_i_ids, dtype=torch.bool)
                    ar_loss_mask = torch.zeros_like(ar_i_ids, dtype=torch.bool)
                    
                    clean_resp_ids = ar_i_ids[start_idx:end_idx]
                    clean_resp_labels = ar_l_ids[start_idx:end_idx]
                    
                    n_blocks = len_res // block_size
                    t_tensor = self._sample_block_times(
                        n_blocks,
                        device=ar_i_ids.device,
                        low=getattr(self, "min_mask_rate", 0.0),
                        high=getattr(self, "max_mask_rate", 1.0),
                    )
                    batch_t.append(t_tensor)

                    diff_i_ids_blocks, diff_l_ids_blocks = [], []
                    diff_t_types_blocks, diff_b_indices_blocks = [], []
                    diff_resp_mask_blocks, diff_loss_mask_blocks = [], [] # 【新增】：Block级Mask容器
                    
                    for b_idx in range(n_blocks):
                        b_start = b_idx * block_size
                        b_end = (b_idx + 1) * block_size
                        
                        b_clean_ids = clean_resp_ids[b_start:b_end]
                        b_clean_labels = clean_resp_labels[b_start:b_end]
                        
                        b_t = t_tensor[b_idx].item()
                        b_mask_prob = 1.0 - b_t
                        
                        b_rand_vals = torch.rand(block_size, device=self.device)
                        b_mask_idx = b_rand_vals < b_mask_prob
                        
                        if b_idx == n_blocks - 1 and pad_len > 0:
                            b_mask_idx[-pad_len:] = False
                        
                        b_noisy_ids = b_clean_ids.clone()
                        b_noisy_ids[b_mask_idx] = mask_token_id
                        
                        # 组装当前 Block 基础属性
                        diff_i_ids_blocks.append(b_noisy_ids)
                        diff_l_ids_blocks.append(b_clean_labels)
                        diff_t_types_blocks.append(torch.full_like(b_noisy_ids, 2))
                        diff_b_indices_blocks.append(torch.full_like(b_noisy_ids, b_idx + 1))
                        
                        # 【新增】：组装当前 Block 的 Masks
                        # response_mask：整个 Block 都是 True
                        diff_resp_mask_blocks.append(torch.ones_like(b_noisy_ids, dtype=torch.bool))
                        # loss_mask：只有被 mask_idx 选中的 token 才是 True
                        diff_loss_mask_blocks.append(b_mask_idx.clone())
                        
                        # 映射 AR 区域对应的 block_index
                        ar_b_indices[start_idx + b_start : start_idx + b_end] = b_idx + 1

                    # 拼装 Diff 区域
                    diff_i_ids = torch.cat(diff_i_ids_blocks)
                    diff_l_ids = torch.cat(diff_l_ids_blocks)
                    diff_t_types = torch.cat(diff_t_types_blocks)
                    diff_b_indices = torch.cat(diff_b_indices_blocks)
                    
                    # 【新增】：拼装 Diff 区域的 Masks
                    diff_response_mask = torch.cat(diff_resp_mask_blocks)
                    diff_loss_mask = torch.cat(diff_loss_mask_blocks)
                    
                    # 单条样本大拼接
                    sample_i_ids = torch.cat([ar_i_ids, diff_i_ids])
                    sample_l_ids = torch.cat([ar_l_ids, diff_l_ids])
                    sample_t_types = torch.cat([ar_t_types, diff_t_types])
                    sample_b_indices = torch.cat([ar_b_indices, diff_b_indices])
                    
                    # 【新增】：单条样本 Masks 拼接
                    sample_resp_mask = torch.cat([ar_response_mask, diff_response_mask])
                    sample_loss_mask = torch.cat([ar_loss_mask, diff_loss_mask])
                    batch_lengths.append((start_idx, len_res))
                else:
                    # 兜底：没有找到回复的情况
                    batch_t.append(torch.empty(0, dtype=torch.float32, device=self.device))
                    sample_i_ids = i_ids[i_ids != pad_token_id] 
                    sample_l_ids = l_ids[:len(sample_i_ids)]
                    sample_t_types = torch.zeros_like(sample_i_ids)
                    sample_b_indices = torch.zeros_like(sample_i_ids)
                    
                    # 【新增】：兜底全为 False
                    sample_resp_mask = torch.zeros_like(sample_i_ids, dtype=torch.bool)
                    sample_loss_mask = torch.zeros_like(sample_i_ids, dtype=torch.bool)
                    batch_lengths.append((len(sample_i_ids), 0))

                # 【新增】：构造 position_ids，让 noisy 段与 clean 段共享位置编码，
                # 消除 RoPE 在 clean/noisy 上的位置偏移，保证训练与推理分布一致。
                sample_pos_ids = torch.arange(sample_i_ids.shape[0], dtype=torch.long)
                if len_res > 0:
                    p_len_i, r_len_i = batch_lengths[-1]
                    sample_pos_ids[p_len_i + r_len_i : p_len_i + 2 * r_len_i] = \
                        torch.arange(p_len_i, p_len_i + r_len_i, dtype=torch.long)

                batch_input_ids.append(sample_i_ids)
                batch_labels.append(sample_l_ids)
                batch_t_types.append(sample_t_types)
                batch_b_indices.append(sample_b_indices)
                batch_response_mask.append(sample_resp_mask) # 【新增】
                batch_loss_mask.append(sample_loss_mask)     # 【新增】
                batch_position_ids.append(sample_pos_ids)    # 【新增】

            # ==========================================
            # Batch 级别的统一 Padding
            # ==========================================
            data_batch['input_ids'] = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
            data_batch['labels'] = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
            data_batch['token_types'] = pad_sequence(batch_t_types, batch_first=True, padding_value=-1)
            data_batch['block_indices'] = pad_sequence(batch_b_indices, batch_first=True, padding_value=-1)
            
            # 【新增】：Padding 值为 False
            data_batch['response_mask'] = pad_sequence(batch_response_mask, batch_first=True, padding_value=False)
            data_batch['loss_mask'] = pad_sequence(batch_loss_mask, batch_first=True, padding_value=False)
            data_batch['t'] = pad_sequence(batch_t, batch_first=True, padding_value=1.0)
            # 【新增】：position_ids padding 用 0，padded 位置本身不参与 attention（被 attention_mask 屏蔽）
            data_batch['position_ids'] = pad_sequence(batch_position_ids, batch_first=True, padding_value=0)
            
            batch_max_len = data_batch['input_ids'].shape[1]
            batch_attn_masks = []
            for p_len, r_len in batch_lengths:
                # 原封不动调用你现有的 self.create_attention_mask 函数
                single_mask = self.create_attention_mask(
                    prefix_len=p_len,
                    res_len=r_len,
                    block_size=block_size,
                    max_len=batch_max_len
                )
                batch_attn_masks.append(single_mask)
                
            # 沿 batch 维度拼接，挂载到 data_batch
            data_batch['attention_mask'] = torch.cat(batch_attn_masks, dim=0).to(self.device)
            
        return data_dict
        
    # ============================================================
    # Block Diffusion Inference (dLLM Generate)
    # ============================================================

    @torch.no_grad()
    def generate_dllm(
        self,
        inputs_embeds,
        max_new_tokens=512,
        block_size=None,
        denoising_steps=None,
        temperature=0.0,
        mask_token_id=151671,
        eos_token_id=151645,
    ):
        """Block Diffusion 推理解码。

        与训练对齐的解码逻辑：
          - 块间 (inter-block): AR 因果，逐块生成并更新 KV Cache。
          - 块内 (intra-block): 迭代去噪，使用全可见注意力；
            每步预测所有 mask token，将置信度最高的固定，其余 remask。

        Args:
            inputs_embeds: [batch, prefix_len, hidden_dim] 已包含图像嵌入的前缀。
            max_new_tokens: 最多生成的 token 数。
            block_size: 每个解码块的大小，默认 self.block_size。
            denoising_steps: 每块去噪步数，默认 block_size（每步固定约 1 个 token）。
            temperature: 采样温度，0 = greedy argmax。
            mask_token_id: Mask 占位符 token ID。
            eos_token_id: 终止 token ID。

        Returns:
            generated_ids: [batch, gen_len] 生成的 token ID 序列（截断到 eos）。
        """
        if block_size is None:
            block_size = getattr(self, 'block_size', 32)
        if denoising_steps is None:
            denoising_steps = block_size

        device = self.device
        dtype = self.dtype
        batch_size = inputs_embeds.shape[0]
        prefix_len = inputs_embeds.shape[1]
        hidden_dim = inputs_embeds.shape[2]

        num_blocks = (max_new_tokens + block_size - 1) // block_size

        # ---------- 1. Prefix 编码，构建初始 KV Cache ----------
        output = self.llm_model(
            inputs_embeds=inputs_embeds,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        kv_cache = output.past_key_values

        # 计算每步去噪应固定的 token 数
        transfer_schedule = self._transfer_schedule(block_size, denoising_steps)

        all_block_ids = []
        finished = False

        # ---------- 2. 逐块生成 ----------
        for b_idx in range(num_blocks):
            # 当前块的 position_ids（与训练一致，contiguous from prefix_len + offset）
            block_start_pos = prefix_len + b_idx * block_size
            block_pos_ids = torch.arange(
                block_start_pos, block_start_pos + block_size,
                device=device, dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

            # 初始化：全部为 mask token
            block_ids = torch.full(
                (batch_size, block_size), mask_token_id,
                dtype=torch.long, device=device,
            )

            # 构造块内全可见 4D attention mask（禁止 HF 默认因果）
            # shape: [batch, 1, block_size, cache_len + block_size]
            cache_len = kv_cache.get_seq_length()
            total_kv_len = cache_len + block_size
            attn_mask_4d = torch.zeros(
                (batch_size, 1, block_size, total_kv_len),
                dtype=dtype, device=device,
            )

            # ---------- 3. 块内迭代去噪 ----------
            for step in range(denoising_steps):
                is_mask = (block_ids == mask_token_id)
                if not is_mask.any():
                    break

                block_embeds = self.llm.get_input_embeddings()(block_ids)

                out = self.llm_model(
                    inputs_embeds=block_embeds,
                    position_ids=block_pos_ids,
                    attention_mask=attn_mask_4d,
                    past_key_values=kv_cache,
                    use_cache=False,
                    return_dict=True,
                )

                # DynamicCache.update() 在 use_cache=False 时仍会扩展 cache。
                # 用公开 API 裁剪并同步 cache 的内部长度记账。
                kv_cache.crop(cache_len)

                logits = self.llm.get_output_embeddings()(
                    out.last_hidden_state
                )  # [batch, block_size, vocab]

                # 采样
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    flat_probs = probs.view(-1, probs.shape[-1])
                    sampled_flat = torch.multinomial(flat_probs, 1)
                    sampled_ids = sampled_flat.view(batch_size, block_size)
                    sampled_conf = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
                else:
                    # Greedy: argmax
                    sampled_conf = F.softmax(logits, dim=-1).max(dim=-1).values
                    sampled_ids = logits.argmax(dim=-1)

                # 仅对仍为 mask 的位置计算置信度
                neg_inf = torch.tensor(float('-inf'), device=device)
                confidence = torch.where(is_mask, sampled_conf, neg_inf)

                # 选择 top-k 置信度最高的 token 固定
                num_to_fix = transfer_schedule[step]
                transfer_mask = torch.zeros_like(block_ids, dtype=torch.bool)
                for j in range(batch_size):
                    n_masked = is_mask[j].sum().item()
                    k = min(num_to_fix, n_masked)
                    if k > 0:
                        _, topk_idx = torch.topk(confidence[j], k)
                        transfer_mask[j, topk_idx] = True

                block_ids = torch.where(transfer_mask, sampled_ids, block_ids)

            # 处理残留 mask（极少情况下最后一步可能仍有 mask）
            residual_mask = (block_ids == mask_token_id)
            if residual_mask.any():
                block_ids[residual_mask] = eos_token_id

            all_block_ids.append(block_ids)

            # 检查是否生成了 eos
            if (block_ids == eos_token_id).any():
                finished = True
                break

            # ---------- 4. Clean forward 更新 KV Cache ----------
            clean_embeds = self.llm.get_input_embeddings()(block_ids)
            clean_out = self.llm_model(
                inputs_embeds=clean_embeds,
                position_ids=block_pos_ids,
                attention_mask=torch.zeros(
                    (batch_size, 1, block_size, total_kv_len),
                    dtype=dtype, device=device,
                ),
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
            kv_cache = clean_out.past_key_values

        # ---------- 5. 拼接并截断 ----------
        generated_ids = torch.cat(all_block_ids, dim=1)

        # 截断到第一个 eos
        for i in range(batch_size):
            eos_pos = (generated_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                generated_ids[i, eos_pos[0] + 1:] = eos_token_id

        return generated_ids

    @staticmethod
    def _transfer_schedule(block_size: int, num_steps: int) -> list:
        """线性调度：每步应固定多少个 token。

        总固定数 = block_size，均匀分配到 num_steps 步；
        最后一步兜底余数，保证总和恰好 = block_size。
        """
        if num_steps >= block_size:
            return [1] * block_size + [0] * (num_steps - block_size)
        base = block_size // num_steps
        remainder = block_size % num_steps
        schedule = []
        for s in range(num_steps):
            schedule.append(base + (1 if s < remainder else 0))
        return schedule

    def compute_loss(self, data_dict):
        if self.dllm: data_dict = self.create_dllm_batch(data_dict)

        losses = {}
        # actual_start_idx, end_idx = self.find_response_span(data_dict['image2text']['input_ids'][0])
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'image2text' in data_type:
                loss = self.image2text_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'depth' in data_type:
                # Depth reconstruction (source image -> target depth map)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'sketch' in data_type:
                # Sketch generation (source image -> target sketch)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'mask' in data_type:
                # Mask segmentation (source image -> target mask)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'joint' in data_type:
                loss_sft, loss_depth = self.joint_loss(batch_data)
                # Weight joint loss parts using individual weights if available, or just sum
                w_sft = self.loss_weights.get('image2text', 1.0)
                w_depth = self.loss_weights.get('edit', 1.0)
                # Use 'joint' weight as global scaler for this branch
                w_joint = self.loss_weights.get('joint', 1.0)
                
                losses[f'loss_{data_type}_sft'] = loss_sft * w_sft * w_joint
                losses[f'loss_{data_type}_depth'] = loss_depth * w_depth * w_joint
            else:
                raise NotImplementedError(f"Unknown data type: {data_type}")
        return losses
