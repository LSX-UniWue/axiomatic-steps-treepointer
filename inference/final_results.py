import argparse
import json
import pathlib
import re

import pandas as pd


pd.options.display.max_columns = 100
drop_columns = [] # ['data_axiom', 'data_position', 'data_step', 'matches', 'equal']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=pathlib.Path)
    parser.add_argument('--max_beam', type=int, default=100, help="highest beam to consider")
    args = parser.parse_args()

    with open("data/trig_multistep_no_reverse_merged/train.json") as f:
        data = json.load(f)

    train_equations = set([d['paren'] for d in data])

    step_files = sorted(args.base_path.glob('*.csv'), key=lambda x: int(re.sub(r'\D', '', str(x))))
    merged = None

    for i, step_file in enumerate(step_files, start=1):
        step_i = pd.read_csv(args.base_path / step_file)
        step_i = step_i[(step_i.beam <= args.max_beam) & (step_i.equal != -1)]
        fails = [f"failed_{x}" for x in range(step_i['transformed_eq'].isna().sum())]
        step_i.loc[step_i.transformed_eq.isna(), 'transformed_eq'] = fails
        step_i['in_training'] = step_i['eq'].isin(train_equations)
        # step_i = clean_equation(step1)
        step_i.drop(columns=drop_columns, inplace=True)
        step_i = step_i.add_suffix(f"_s{i}")
        if merged is None:
            merged = step_i
        else:
            merged = pd.merge(merged, step_i, how="left", left_on=f"transformed_eq_s{i-1}", right_on=f"eq_s{i}")

    finished_columns = [c for c in merged.columns if c.startswith('finished_')]
    in_training_columns = [c for c in merged.columns if c.startswith('in_training_')]

    merged['total_finished'] = merged[finished_columns].max(axis=1)
    grouped_eq = merged[['id_s1'] + finished_columns + ["total_finished"]].groupby("id_s1")
    finished = grouped_eq.max()[finished_columns + ["total_finished"]]
    finished_with_dist = pd.merge(finished, merged[['id_s1', 'dist_s1'] + in_training_columns], left_index=True, right_on="id_s1", how="left").drop_duplicates(subset="id_s1").fillna(0)
    finished_with_dist['in_training_count'] = merged[in_training_columns].sum(axis=1)

    print(finished_with_dist.fillna(0).groupby('dist_s1').mean())
    merged.to_csv(args.base_path / 'merged_frame.csv', index=False)
    finished_with_dist.fillna(0).groupby('dist_s1').mean().to_csv(args.base_path / 'final_results.csv', index=False)
