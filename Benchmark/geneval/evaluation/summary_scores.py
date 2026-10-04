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

import argparse
import os

import pandas as pd


def summarize_results(results):
    frame = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    scores = frame.groupby('tag')['correct'].mean()
    return float(scores.mean() * 100), {tag: float(score * 100) for tag, score in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default='/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/public/tupeng/UniMRG/Benchmark/results_UniMRG_gen_bs16_iter4000.jsonl')
    args = parser.parse_args()

    # Load results

    df = pd.read_json(args.filename, orient="records", lines=True)

    # Measure overall success

    print("Summary")
    print("=======")
    print(f"Total images: {len(df)}")
    print(f"Total prompts: {len(df.groupby('metadata'))}")
    print(f"% correct images: {df['correct'].mean():.2%}")
    print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
    print()

    # By group

    overall, metrics = summarize_results(df)
    task_scores = []

    print("Task breakdown")
    print("==============")
    for tag, task_df in df.groupby('tag', sort=False):
        task_scores.append(metrics[tag] / 100)
        print(f"{tag:<16} = {metrics[tag] / 100:.2%} ({task_df['correct'].sum()} / {len(task_df)})")
    print()

    print(f"Overall score (avg. over tasks): {overall / 100:.5f}")


    print("\n\n==============")
    output_info = "SO   TO   CT   CL   POS  ATTR ALL\n"
    for score in task_scores:
        output_info += f"{score:.2f} "
    output_info += f"{overall / 100:.2f}" + "\n"
    print(output_info)
    with open(os.path.join(os.path.dirname(args.filename), "geneval_results.txt"), "w") as f:
        f.write(output_info)


if __name__ == '__main__':
    main()
