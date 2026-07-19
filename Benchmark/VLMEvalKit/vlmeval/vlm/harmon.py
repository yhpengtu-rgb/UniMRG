import os
import sys
import torch
import warnings
import numpy as np
from PIL import Image
from einops import rearrange
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class Harmon(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False  # Harmon implementation in utils.py handles single image

    def check_install(self):
        """Check if Harmon dependencies are installed."""
        try:
            # Add Harmon repo root to sys.path (sibling folder named ``Harmon`` next to VLMEvalKit).
            harmon_path = os.path.join(os.path.dirname(__file__), '../../../../Harmon')
            if os.path.exists(harmon_path) and harmon_path not in sys.path:
                sys.path.insert(0, harmon_path)
            
            import mmengine
            import xtuner
            from src.builder import BUILDER
        except Exception as e:
            logging.critical(
                'Please ensure Harmon and its dependencies (mmengine, xtuner) are available.')
            raise e

    def __init__(self, model_path=None, checkpoint_path=None, image_size=512,
                 use_dllm=False, block_size=32, denoising_steps=None, **kwargs):
        """
        Initialize Harmon model for VLM evaluation.
        
        Args:
            model_path: Path to model config file (Config)
            checkpoint_path: Path to model checkpoint (Checkpoint)
            image_size: Input image size (default: 512)
            use_dllm: If True, use block diffusion decoding instead of AR.
            block_size: Block size for dLLM decoding.
            denoising_steps: Number of denoising iterations per block (default=block_size).
            **kwargs: Additional generation kwargs
        """
        self.check_install()
        
        from mmengine.config import Config
        from src.builder import BUILDER
        from xtuner.model.utils import guess_load_checkpoint
        
        assert model_path is not None, "model_path (config) must be provided"
        assert checkpoint_path is not None, "checkpoint_path must be provided"
        
        self.config_path = model_path
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.use_dllm = use_dllm
        self.dllm_block_size = block_size
        self.dllm_denoising_steps = denoising_steps
        
        # Load Config
        config = Config.fromfile(self.config_path)
        
        # If using dLLM decoding, ensure model is built with dllm=True
        if self.use_dllm:
            config.model['dllm'] = True
            config.model['block_size'] = block_size
        
        # Build Model
        print(f"Building Harmon model from {self.config_path}...")
        model = BUILDER.build(config.model).eval().cuda()
        model = model.to(model.dtype)
        
        # Load Checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = self._load_checkpoint(self.checkpoint_path)
            
        info = model.load_state_dict(checkpoint, strict=False)
        
        # Filter out expected VAE missing keys
        unexpected_missing_keys = [k for k in info.missing_keys if not k.startswith('vae.')]
        if unexpected_missing_keys or info.unexpected_keys:
             print(f"Checkpoint loaded with unexpected issues: {info}")
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
        num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        
        # Verify prompt template
        if not hasattr(model, 'prompt_template') or model.prompt_template is None:
            raise ValueError("Model does not have prompt_template attribute. Check model config.")
        if 'INSTRUCTION' not in model.prompt_template:
            raise ValueError("Model prompt_template does not have 'INSTRUCTION' key. Check model config.")
            
        self.model = model
        
        # Set default generation kwargs
        default_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        
        mode_str = 'dLLM (block diffusion)' if self.use_dllm else 'AR (autoregressive)'
        warnings.warn(f'Harmon model loaded [{mode_str}]. Generation kwargs: {self.kwargs}')

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        import os
        import torch
        from xtuner.model.utils import guess_load_checkpoint

        if os.path.isdir(checkpoint_path):
            # 检查是否是 DeepSpeed ZeRO 目录
            model_states = os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt')
            if os.path.isfile(model_states):
                print(f"[Harmon] 检测到 DeepSpeed ZeRO-2 目录，从 {model_states} 加载模型权重")
                ck = torch.load(model_states, map_location='cpu', weights_only=False)
                if 'module' in ck:
                    return ck['module']
                raise KeyError(f"'module' key not found in {model_states}. Keys: {list(ck.keys())}")
            # 尝试 xtuner 的 guess_load_checkpoint
            return guess_load_checkpoint(checkpoint_path)
        else:
            # 单文件：尝试 mmengine 格式（state_dict key），再尝试原生
            try:
                ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                ck = torch.load(checkpoint_path, map_location='cpu')
            # 常见 wrapper key
            for key in ('state_dict', 'module', 'model'):
                if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
                    print(f"[Harmon] 从 checkpoint['{key}'] 提取权重")
                    return ck[key]
            return ck

    def expand2square(self, pil_img, background_color=(127, 127, 127)):
        """Expand image to square by padding"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_image(self, image):
        """Process image for Harmon model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.expand2square(image, (127, 127, 127))
        image = image.resize(size=(self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=self.model.dtype, device=self.model.device)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1
        return image

    def generate_inner(self, message, dataset=None):
        """
        Generate response using Harmon model.
        Supports both AR (autoregressive) and dLLM (block diffusion) decoding.
        """
        prompt, image_path = self.message_to_promptimg(message)

        if image_path is None:
            return "Error: Image required for Harmon evaluation."

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = (
                f"{prompt.rstrip()}\n"
                "Answer with the option's letter from the given choices directly."
            )

        # Process Image
        image_tensor = self.process_image(image_path)
        
        # Prepare Prompt
        prompt_template = self.model.prompt_template['INSTRUCTION']
        formatted_prompt = prompt_template.format(input="<image>\n" + prompt)
        
        # Replace <image> with multiple image tokens
        image_length = (self.image_size // 16) ** 2 + 64
        formatted_prompt = formatted_prompt.replace('<image>', '<image>' * image_length)
        
        # Tokenize
        input_ids = self.model.tokenizer.encode(
            formatted_prompt, add_special_tokens=True, return_tensors='pt').cuda()
            
        # Extract visual features
        with torch.no_grad():
            _, z_enc = self.model.extract_visual_feature(self.model.encode(image_tensor))
            
        # Create inputs embeddings
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
        inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
        
        # Fill in text embeddings
        mask_text = input_ids != self.image_token_idx
        inputs_embeds[mask_text] = self.model.llm.get_input_embeddings()(input_ids[mask_text])
        
        # Generate
        with torch.no_grad():
            if self.use_dllm:
                # Block Diffusion decoding
                generated_ids = self.model.generate_dllm(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=self.kwargs.get('max_new_tokens', 512),
                    block_size=self.dllm_block_size,
                    denoising_steps=self.dllm_denoising_steps,
                    temperature=self.kwargs.get('temperature', 0.0),
                )
                # Decode generated token IDs
                response = self.model.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
            else:
                # Standard AR decoding
                output = self.model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    do_sample=self.kwargs.get('do_sample', False),
                    max_new_tokens=self.kwargs.get('max_new_tokens', 1024),
                    eos_token_id=self.model.tokenizer.eos_token_id,
                    pad_token_id=self.model.tokenizer.pad_token_id 
                    if self.model.tokenizer.pad_token_id is not None else 
                    self.model.tokenizer.eos_token_id,
                    temperature=self.kwargs.get('temperature', 1.0) if self.kwargs.get('do_sample', False) else None
                )
                response = self.model.tokenizer.decode(output[0], skip_special_tokens=True)

        response = response.strip()
        if response.endswith('.'):
            response = response[:-1]

        return response

    def use_custom_prompt(self, dataset):
        return False
