import os
import sys
import torch
import warnings
import numpy as np
from collections.abc import Mapping
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
                 use_dllm=False, block_size=32, denoising_steps=None,
                 preflight_samples=0, preflight_require_eos=True,
                 preflight_require_nonempty=True, **kwargs):
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
        self._preflight_remaining = int(preflight_samples)
        if self._preflight_remaining < 0:
            raise ValueError('preflight_samples must be non-negative')
        self.preflight_require_eos = bool(preflight_require_eos)
        self.preflight_require_nonempty = bool(preflight_require_nonempty)
        self.runtime_stats_path = (
            os.environ.get('HARMON_RUNTIME_STATS_PATH')
            if self.use_dllm else None
        )
        
        # Load Config
        config = Config.fromfile(self.config_path)
        
        # If using dLLM decoding, ensure model is built with dllm=True
        if self.use_dllm:
            config.model['dllm'] = True
            config.model['block_size'] = block_size
        
        # Build Model
        print(f"Building Harmon model from {self.config_path}...")
        if 'LOCAL_RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        model = BUILDER.build(config.model).eval().cuda()
        model = model.to(model.dtype)
        
        # Load Checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = self._load_checkpoint(self.checkpoint_path)
        checkpoint_audit = self._audit_checkpoint_keys(
            model,
            checkpoint,
            required_key_groups=tuple(config.get(
                'checkpoint_required_key_groups', ())),
        )
        print(
            '[Harmon] checkpoint key audit PASS: '
            f"matched={checkpoint_audit['matched_keys']}/"
            f"{checkpoint_audit['checkpoint_keys']}, "
            f"required={checkpoint_audit['required_key_groups']}")

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
    def _audit_checkpoint_keys(
        model,
        checkpoint,
        *,
        required_key_groups=(),
    ):
        """Fail closed when evaluation would silently skip model weights."""
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                'evaluation checkpoint must be a state_dict mapping, got '
                f'{type(checkpoint).__name__}')

        model_keys = set(model.state_dict())
        checkpoint_keys = set(checkpoint)
        matched_keys = model_keys & checkpoint_keys
        if not matched_keys:
            raise RuntimeError(
                'evaluation checkpoint matched 0 model keys; refusing to '
                'run an uninitialized model')

        group_counts = {}
        for marker in required_key_groups:
            if not isinstance(marker, str) or not marker:
                raise ValueError(
                    'checkpoint_required_key_groups entries must be '
                    f'non-empty strings, got {marker!r}')
            model_group = {key for key in model_keys if marker in key}
            # Required groups describe model parameters that must be restored.
            # A full training checkpoint may legitimately contain additional
            # modules which are not constructed for evaluation (for example
            # ``vae.*.proj_out.*``).  Restrict the checkpoint side to keys that
            # can actually match this evaluation model; otherwise a generic
            # marker such as ``proj_out.`` falsely treats those extra modules
            # as an incomplete top-level projection group.
            checkpoint_group = {
                key for key in matched_keys if marker in key
            }
            if not model_group:
                raise RuntimeError(
                    f'evaluation model has no keys for required group '
                    f'{marker!r}')
            if checkpoint_group != model_group:
                missing = sorted(model_group - checkpoint_group)[:3]
                unexpected = sorted(checkpoint_group - model_group)[:3]
                raise RuntimeError(
                    f'evaluation checkpoint required group {marker!r} is '
                    f'incomplete: matched={len(checkpoint_group & model_group)}/'
                    f'{len(model_group)}, missing_sample={missing}, '
                    f'unexpected_sample={unexpected}')
            group_counts[marker] = len(model_group)

        return {
            'checkpoint_keys': len(checkpoint_keys),
            'matched_keys': len(matched_keys),
            'required_key_groups': group_counts,
        }

    def _validate_dllm_preflight(self, generated_ids, response, dataset):
        """Fail early when a checkpoint cannot produce a valid first answer."""
        if self._preflight_remaining <= 0:
            return

        eos_token_id = self.model.tokenizer.eos_token_id
        if self.preflight_require_eos:
            if eos_token_id is None:
                raise SystemExit(
                    'dLLM evaluation preflight requires EOS, but the '
                    'tokenizer has no eos_token_id')
            emitted_eos = bool(
                (generated_ids == int(eos_token_id)).any().item())
            if not emitted_eos:
                preview = generated_ids[0, :16].detach().cpu().tolist()
                raise SystemExit(
                    'dLLM evaluation preflight did not emit EOS for '
                    f'dataset={dataset!r}; first token ids={preview}. '
                    'Stop: the checkpoint or decoding contract is invalid.')

        if self.preflight_require_nonempty and not response.strip():
            raise SystemExit(
                'dLLM evaluation preflight decoded to an empty response for '
                f'dataset={dataset!r}. Stop: do not continue formal testing.')

        self._preflight_remaining -= 1
        print(
            '[Harmon] dLLM evaluation preflight PASS: '
            f'dataset={dataset!r}, remaining={self._preflight_remaining}')

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
                    sd = ck[key]
                    # DDP wrapper 给所有 key 加了 'module.' 前缀，推理时模型不是 DDP，
                    # 必须 strip 掉否则 load_state_dict(strict=False) 会静默跳过所有参数。
                    if any(k.startswith('module.') for k in sd):
                        print(f"[Harmon] 检测到 DDP 'module.' 前缀，strip {sum(k.startswith('module.') for k in sd)}/{len(sd)} keys")
                        sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
                    return sd
            # Flat state_dict (no wrapper key) — still may carry the DDP
            # 'module.' prefix from training.  Strip it here as well,
            # otherwise ``model.load_state_dict(strict=False)`` silently
            # skips every parameter (memory: "strict=False causes silent
            # failure when keys don't match"), leaving LoRA / RankGate /
            # RiskHead at their random init and producing garbage outputs
            # (e.g. MMBench 92% NaN predictions).
            if isinstance(ck, dict) and any(
                    k.startswith('module.') for k in ck):
                print(f"[Harmon] 检测到 DDP 'module.' 前缀 (flat state_dict)，"
                      f"strip {sum(k.startswith('module.') for k in ck)}/{len(ck)} keys")
                ck = {k[len('module.'):] if k.startswith('module.') else k: v
                      for k, v in ck.items()}
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
                # Block Diffusion decoding.
                # TRACER-LoRA factorial switches. Both cells still build the
                # same modules and execute the same RiskHead; the switches
                # only decide whether its lagged output is applied to the
                # RankGate and/or committed-set revision.
                import os as _os

                def _env_bool(name, default):
                    raw = _os.environ.get(name)
                    if raw is None:
                        return bool(default)
                    value = raw.strip().lower()
                    if value in ('1', 'true', 'yes'):
                        return True
                    if value in ('0', 'false', 'no'):
                        return False
                    raise ValueError(
                        f'{name} must be true/false, got {raw!r}')

                tracer_router_enabled = _env_bool(
                    'TRACER_ROUTER_ENABLED', True)
                # Legacy GUARD scripts remain reproducible: when the new
                # policy variable is absent, honour GUARD_RISK_COMMIT.
                legacy_policy = _env_bool('GUARD_RISK_COMMIT', True)
                tracer_policy_enabled = _env_bool(
                    'TRACER_POLICY_ENABLED', legacy_policy)
                runtime_diagnostics = None
                if self.runtime_stats_path:
                    from src.models.dllm.guard.runtime_diagnostics import (
                        TracerRuntimeDiagnostics,
                    )
                    resolved_steps = (
                        self.dllm_denoising_steps
                        if self.dllm_denoising_steps is not None
                        else self.dllm_block_size
                    )
                    runtime_diagnostics = TracerRuntimeDiagnostics(
                        router_enabled=tracer_router_enabled,
                        policy_enabled=tracer_policy_enabled,
                        block_size=self.dllm_block_size,
                        denoising_steps=resolved_steps,
                        max_new_tokens=self.kwargs.get(
                            'max_new_tokens', 512),
                    )
                from contextlib import nullcontext
                binding_mode = _os.environ.get('TRACER_BINDING_MODE', 'default')
                if binding_mode not in ('default', 'policy', 'no_revision', 'token_matched'):
                    raise ValueError(f'Invalid TRACER_BINDING_MODE: {binding_mode}')
                binding_context = nullcontext()
                if binding_mode != 'default':
                    if tracer_router_enabled or not tracer_policy_enabled:
                        raise ValueError('Binding controls require router off and policy on')
                    from scripts.tracer_binding_policy import binding_policy
                    binding_context = binding_policy(
                        binding_mode, int(self.model.harmon_mask_token_id))
                with binding_context:
                    generated_ids = self.model.generate_dllm(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=self.kwargs.get('max_new_tokens', 512),
                        block_size=self.dllm_block_size,
                        denoising_steps=self.dllm_denoising_steps,
                        temperature=self.kwargs.get('temperature', 0.0),
                        tracer_router_enabled=tracer_router_enabled,
                        tracer_policy_enabled=tracer_policy_enabled,
                        runtime_diagnostics=runtime_diagnostics,
                    )
                if (
                    runtime_diagnostics is not None
                    and (
                        generated_ids.ndim != 2
                        or generated_ids.shape[0] != 1
                    )
                ):
                    raise RuntimeError(
                        'Harmon runtime diagnostics require evaluation '
                        'batch size 1')
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

        if self.use_dllm:
            if runtime_diagnostics is not None:
                from src.models.dllm.guard.runtime_diagnostics import (
                    append_runtime_record,
                )
                runtime_record = runtime_diagnostics.finalize(
                    generated_ids,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                )
                runtime_record.update({
                    'dataset': dataset or 'unknown',
                    'response_chars': len(response),
                })
                append_runtime_record(
                    self.runtime_stats_path, runtime_record)
            self._validate_dllm_preflight(
                generated_ids=generated_ids,
                response=response,
                dataset=dataset,
            )

        return response

    def use_custom_prompt(self, dataset):
        return False
