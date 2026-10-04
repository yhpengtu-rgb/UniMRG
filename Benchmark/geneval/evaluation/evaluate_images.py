# Copyright (c) 2023 Dhruba Ghosh
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/djghosh13/geneval/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""
Evaluate generated images using Mask2Former (or other object detector model)
"""

import argparse
import json
import os
import re
import sys
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

# Get directory path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, default='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/tupeng/UniMRG/Harmon/results/UniMRG_gen_bs16_iter4000')
    parser.add_argument("--outfile", type=str, default="results_UniMRG_gen_bs16_iter4000.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/tupeng/UniMRG/model/")
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            ".mim/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
            if int(mmdet.__version__.split('.')[0]) >= 3 else
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
    return args

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

# Load models

@timed
def load_models(args):
    CONFIG_PATH = args.model_config
    OBJECT_DETECTOR = args.options.get('model', "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    CKPT_PATH = os.path.join(args.model_path, f"{OBJECT_DETECTOR}.pth")
    if int(mmdet.__version__.split('.')[0]) >= 3:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        weights = checkpoint['state_dict']
        # MMDetection renamed transformer submodules in 3.x. Attention 0 in
        # the Mask2Former decoder is cross attention; attention 1 is self attention.
        converted = {}
        for key, value in weights.items():
            if 'pixel_decoder.encoder.layers.' in key:
                key = key.replace('.attentions.0.', '.self_attn.')
            elif 'transformer_decoder.layers.' in key:
                key = key.replace('.attentions.0.', '.cross_attn.').replace('.attentions.1.', '.self_attn.')
            key = key.replace('.ffns.0.', '.ffn.')
            converted[key] = value
        object_detector = init_detector(CONFIG_PATH, None, device=DEVICE)
        # Reject incomplete loading rather than scoring with random parameters.
        object_detector.load_state_dict(converted, strict=True)
        print('Detector weights loaded completely (MMDetection 3.x compatibility).', file=sys.stderr)
    else:
        object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = args.options.get('clip_model', "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}

# Evaluation parts

class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        bgcolor = args.options.get('bgcolor', "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if args.options.get('crop', '1') == '1':
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (transform(image), 0)


def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes),
        batch_size=16, num_workers=4
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                # Color check
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                # Relative position check
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    if hasattr(result, 'pred_instances'):  # MMDetection 3.x
        instances = result.pred_instances.cpu()
        labels = instances.labels.numpy()
        boxes = np.concatenate([instances.bboxes.numpy(), instances.scores.numpy()[:, None]], axis=1)
        bbox = [boxes[labels == index] for index in range(len(classnames))]
        masks = instances.masks.numpy() if 'masks' in instances else None
        segm = [masks[labels == index] for index in range(len(classnames))] if masks is not None else None
    else:  # MMDetection 2.x
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist() # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


def main(args):
    # Persist each image so interrupted scoring can resume without repeating inference.
    cached = {}
    if os.path.isfile(args.outfile):
        with open(args.outfile) as stream:
            lines = stream.readlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                continue
            cached[row['filename']] = row
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, 'w') as stream:
        for row in cached.values():
            stream.write(json.dumps(row) + '\n')
    for subfolder in tqdm(sorted(os.listdir(args.imagedir))):
        folderpath = os.path.join(args.imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        # Evaluate each image
        for imagename in sorted(os.listdir(os.path.join(folderpath, "samples"))):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            stat = os.stat(imagepath)
            stamp = [stat.st_size, stat.st_mtime_ns]
            previous = cached.get(imagepath)
            if previous and previous.get('image_stamp') == stamp and json.loads(previous['metadata']) == metadata:
                continue
            result = evaluate_image(imagepath, metadata)
            result['image_stamp'] = stamp
            cached[imagepath] = result
            with open(args.outfile, 'a') as fp:
                fp.write(json.dumps(result) + '\n')
    # Drop stale entries and earlier versions of overwritten images.
    current = {os.path.join(args.imagedir, folder, 'samples', filename)
               for folder in os.listdir(args.imagedir) if folder.isdigit()
               for filename in os.listdir(os.path.join(args.imagedir, folder, 'samples'))
               if re.fullmatch(r'\d+\.png', filename)}
    with open(args.outfile, 'w') as stream:
        for name, result in cached.items():
            if name in current:
                stream.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    args = parse_args()
    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    THRESHOLD = float(args.options.get('threshold', 0.3))
    COUNTING_THRESHOLD = float(args.options.get('counting_threshold', 0.9))
    MAX_OBJECTS = int(args.options.get('max_objects', 16))
    NMS_THRESHOLD = float(args.options.get('max_overlap', 1.0))
    POSITION_THRESHOLD = float(args.options.get('position_threshold', 0.1))

    main(args)
