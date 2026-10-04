# coding=utf-8
# Copyright 2024 Harmon Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import numpy as np
import torch
from geneval import sample_images
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange
from tqdm import tqdm, trange
import json

from xtuner.model.utils import guess_load_checkpoint

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/harmon_1.5b.pth",
        help="Path to Harmon checkpoint (.pth or checkpoint directory).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, default="t2i")
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=3.0)
    parser.add_argument("--generation_timesteps", "--num_iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--cfg_prompt', type=str, default='Generate an image.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--outdir', type=str, default='dpg_harmon_ori')
    parser.add_argument('--prompts_file', type=str,
                        default='prompts/dpgbench/prompts.json')
    parser.add_argument('--l', type=int, default=0, help='Start index for processing')
    parser.add_argument('--r', type=int, default=None, help='End index for processing')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = Config.fromfile(args.config)
    
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
    info = model.load_state_dict(checkpoint, strict=False)
    
    try:
        with open(args.prompts_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        dataset = {"default.txt": "a dog on the left and a cat on the right."}

    l = args.l
    r = args.r if args.r is not None else len(dataset)
    
    # Prompts for this rank/process: slice [l, r) of the full list.
    all_items = list(dataset.items())
    assigned_items = all_items[l:r]
    total_items = len(dataset)
    
    print(f"This GPU will process {len(assigned_items)} prompts (index {l} to {r-1} out of {total_items} total)")
    
    for idx, (key, prompt) in enumerate(tqdm(assigned_items, desc="Processing prompts")):
        global_index = l + idx  # index in the full dataset
            
        os.makedirs(args.outdir, exist_ok=True)

        print(f"Prompt ({global_index+1}/{total_items}, key={key}): '{prompt}'")

        # Skip if every output image for this prompt already exists.
        batch_size = args.batch_size
        all_exist = True
        for img_idx in range(batch_size):
            out_path = os.path.join(args.outdir, f"{key.split('.')[-2]}_{img_idx}.jpg")
            if not os.path.exists(out_path):
                all_exist = False
                break
        
        if all_exist:
            print(f"Skipping generation for {key} - all images already exist")
            continue
            
        full_prompt = f"Generate an image: {prompt}"
        class_info = model.prepare_text_conditions(full_prompt, args.cfg_prompt)
        
        with torch.no_grad():
            samples = sample_images(
                model, class_info, batch_size, image_size=args.image_size,
                num_iter=args.generation_timesteps, cfg=args.guidance_scale,
                cfg_schedule=args.cfg_schedule, temperature=args.temperature,
                progress=True)

        for idx, sample in enumerate(samples):
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            sample = sample.transpose(1, 2, 0)
            out_path = os.path.join(args.outdir, f"{key.split('.')[-2]}_{idx}.jpg")
            Image.fromarray(sample).save(out_path)
    
    print("Done!")
