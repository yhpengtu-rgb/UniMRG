import argparse
import csv
import json
from pathlib import Path
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default='./dpg_bench/dpg_bench.csv',
    )
    parser.add_argument(
        "--res-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pic-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqa-model",
        type=str,
        default='mplug',
    )

    args = parser.parse_args()
    return args


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu'):
        super().__init__()
        os.environ.setdefault('MODELSCOPE_CACHE', '/nvmedata/xiexu/data/models/modelscope')
        cached_model = Path(os.environ['MODELSCOPE_CACHE']) / ckpt
        if not Path(ckpt).is_dir() and (cached_model / 'pytorch_model.bin').is_file():
            ckpt = str(cached_model)
        if not Path(ckpt).is_dir():
            from modelscope.hub.snapshot_download import snapshot_download
            ckpt = snapshot_download(ckpt, revision='v1.0.0', cache_dir=os.environ['MODELSCOPE_CACHE'])
        from modelscope.models.multi_modal.mplug import MPlug
        from torchvision import transforms
        from transformers import BertTokenizer
        self.model = MPlug.from_pretrained(ckpt, load_checkpoint=False)
        weights = torch.load(Path(ckpt) / 'pytorch_model.bin', map_location='cpu', weights_only=False)
        weights = weights.get('model', weights)
        weights = weights.get('module', weights)
        weights = {k.replace('model.', ''): v for k, v in weights.items()}
        incompatible = self.model.load_state_dict(weights, strict=False)
        missing_parameters = set(incompatible.missing_keys) & set(dict(self.model.named_parameters()))
        if missing_parameters:
            raise RuntimeError(f'Missing mPLUG parameters: {sorted(missing_parameters)}')
        print(f'mPLUG checkpoint loaded: {incompatible}', flush=True)
        self.device = torch.device('cuda:0' if device == 'gpu' else device)
        self.model.to(self.device).eval()
        self.tokenizer = BertTokenizer.from_pretrained(ckpt)
        # Match ModelScope MPlugPreprocessor: bicubic resize, CLIP normalize,
        # lowercase questions, 25-token padding/truncation, and unchanged decoding.
        self.transform = transforms.Compose([
            transforms.Resize((self.model.config.image_res, self.model.config.image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.last_image = None
        self.visual_features = None
        self.cache_visual = True
        visual_forward = self.model.visual_encoder.visual.forward

        def cached_visual(*args, **kwargs):
            if not self.cache_visual:
                return visual_forward(*args, **kwargs)
            if self.visual_features is None:
                self.visual_features = visual_forward(*args, **kwargs)
            return self.visual_features.clone()

        self.model.visual_encoder.visual.forward = cached_visual

    def vqa(self, image, question):
        with torch.inference_mode():
            if image is not self.last_image:
                self.image_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
                self.last_image = image
                self.visual_features = None
            tokens = self.tokenizer(question.lower(), padding='max_length', truncation=True,
                                    max_length=25, return_tensors='pt').to(self.device)
            predictions, _ = self.model(self.image_tensor, tokens, train=False)
            return self.tokenizer.decode(predictions[0][0], skip_special_tokens=True)


def dpg_questions(path):
    questions = defaultdict(dict)
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            questions[row['item_id']][int(row['proposition_id'])] = {
                'question': row['question_natural_language'],
                'parents': [int(x.strip()) for x in row['dependency'].split(',')],
                'category': row['tuple'].split('(')[0].strip(),
            }
    return questions


def score_dpg_image(image, questions, model):
    # Separate generated files are individual images, not four-panel grids.
    raw = {qid: float(model.vqa(image, q['question']) == 'yes')
           for qid, q in questions.items()}
    scores = raw.copy()
    # Preserve the bundled evaluator's dependency processing order.
    for qid, question in questions.items():
        if any(scores[parent] == 0 for parent in question['parents'] if parent != 0):
            scores[qid] = 0.0
    return {
        'score': sum(scores.values()) / len(scores),
        'questions': [{'category': q['category'], 'raw_score': raw[qid],
                       'score': scores[qid]} for qid, q in questions.items()],
    }


def score_images(images, csv_path, details_path, model_id='damo/mplug_visual-question-answering_coco_large_en', vqa=None):
    """Score individual files once; resume cached results by image size/mtime."""
    questions = dpg_questions(csv_path)
    for image in images:
        if image.stem.rsplit('_', 1)[0] not in questions:
            raise ValueError(f'CSV has no questions for {image.name}')
    details_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if details_path.exists():
        lines = details_path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                continue  # A last line may be truncated by interruption.
            cached[result['filename']] = result
        details_path.write_text(''.join(json.dumps(x) + '\n' for x in cached.values()))
    results = []
    with details_path.open('a', encoding='utf-8') as stream:
        for i, image in enumerate(images):
            stamp = [image.stat().st_size, image.stat().st_mtime_ns]
            result = cached.get(str(image))
            if result is None or result.get('image_stamp') != stamp:
                if vqa is None:
                    vqa = MPLUG(ckpt=model_id, device='cuda:0')
                with Image.open(image) as img:
                    result = score_dpg_image(img.convert('RGB'), questions[image.stem.rsplit('_', 1)[0]], vqa)
                result.update(filename=str(image), image_stamp=stamp)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
            results.append(result)
            if (i + 1) % 20 == 0 or i + 1 == len(images):
                print(f'Scored {i+1}/{len(images)}', flush=True)
    return results


def category_scores(results):
    groups = defaultdict(list)
    for result in results:
        for question in result['questions']:
            groups[question['category']].append(question['raw_score'])
    return groups


def summarize_results(results):
    groups = defaultdict(list)
    for category, values in category_scores(results).items():
        groups[category.split('-')[0].strip()].extend(values)
    return (sum(r['score'] for r in results) / len(results) * 100,
            {tag: sum(values) / len(values) * 100 for tag, values in sorted(groups.items())})


def main():
    args = parse_args()

    accelerator = Accelerator()

    try:

        timestamp = time.time()
        time_array = time.localtime(timestamp)
        time_style = time.strftime("%Y%m%d-%H%M%S", time_array)
        if args.res_path is None:
            args.res_path = osp.join(args.image_root_path, f'dpg-bench_{time_style}_results.txt')
        if accelerator.is_main_process:
            with open(args.res_path, 'w') as f:
                pass

        device = str(accelerator.device)
        print(f"进程 {accelerator.process_index} 使用设备: {device}")
        
        # 只让主进程下载和初始化模型，其他进程等待
        if accelerator.is_main_process:
            print("主进程正在下载和初始化模型...")
            if args.vqa_model == 'mplug':
                vqa_model = MPLUG(device=device)
            else:
                raise NotImplementedError(f"VQA 模型 {args.vqa_model} 未实现")
        
        # 等待主进程完成模型下载
        accelerator.wait_for_everyone()
        
        # 其他进程现在可以安全地初始化模型
        if not accelerator.is_main_process:
            print(f"进程 {accelerator.process_index} 正在初始化模型...")
            if args.vqa_model == 'mplug':
                vqa_model = MPLUG(device=device)
            else:
                raise NotImplementedError(f"VQA 模型 {args.vqa_model} 未实现")
            
        vqa_model = accelerator.prepare(vqa_model)
        vqa_model = getattr(vqa_model, 'module', vqa_model) 

        filename_list = sorted(fn for fn in os.listdir(args.image_root_path) if Path(fn).suffix.lower() in {'.png', '.jpg', '.jpeg'})
        num_each_rank = len(filename_list) / accelerator.num_processes
        local_rank = accelerator.process_index
        local_filename_list = filename_list[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

        model_id = osp.basename(args.image_root_path)
        print(f'进程 {local_rank} 开始评估 {model_id}，处理 {len(local_filename_list)} 个文件')
        
        results = score_images(
            [Path(args.image_root_path) / fn for fn in local_filename_list],
            Path(args.csv), Path(args.res_path).with_suffix(f'.rank{local_rank}.jsonl'),
            vqa=vqa_model)
        local_scores = [result['score'] for result in results]
        local_category2scores = category_scores(results)

        accelerator.wait_for_everyone()
        global_dpg_scores = gather_object(local_scores)
        mean_dpg_score = np.mean(global_dpg_scores)

        global_categories = gather_object(list(local_category2scores.keys()))
        global_categories = set(global_categories)
        global_category2scores = dict()
        for category in sorted(global_categories):
            local_category_scores = local_category2scores.get(category, [])
            global_category2scores[category] = gather_object(local_category_scores)
        
        global_category2scores_l1 = defaultdict(list)
        for category in sorted(global_categories):
            l1_category = category.split('-')[0].strip()
            global_category2scores_l1[l1_category].extend(global_category2scores[category])

        if accelerator.is_main_process:
            output = f'Model: {model_id}\n'
            
            output += 'L1 category scores:\n'
            for l1_category in global_category2scores_l1.keys():
                output += f'\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n'
            
            output += 'L2 category scores:\n'
            for category in sorted(global_categories):
                output += f'\t{category}: {np.mean(global_category2scores[category]) * 100}\n'

            output += f'Image path: {args.image_root_path}\n'
            output += f'Save results to: {args.res_path}\n'
            output += f'DPG-Bench score: {mean_dpg_score * 100}'
            with open(args.res_path, 'a') as f:
                f.write(output + '\n')
            print(output)

    except Exception as e:
        print(f"进程 {accelerator.process_index} 发生错误: {e}")
        raise
    
    finally:
        # 确保正确清理分布式进程组
        print(f"进程 {accelerator.process_index} 正在清理资源...")
        try:
            # 等待所有进程完成
            accelerator.wait_for_everyone()
            
            # 清理 accelerator
            del accelerator
            
            # 如果使用了 torch.distributed，清理进程组
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
        except Exception as cleanup_error:
            print(f"进程清理时出现警告: {cleanup_error}")


if __name__ == "__main__":
    main()
