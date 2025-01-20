import argparse
import pathlib
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help="The results of the previous step as .csv")
    parser.add_argument('step', help="The step to write")
    parser.add_argument('--distance', type=int, default=2, help='The minimal distance to consider')
    parser.add_argument('--remove_paren', action='store_true', help='Remove parentheses for the flat pointer')
    args = parser.parse_args()
    results_file = pathlib.Path(args.results)
    outpath = results_file.parent / f"data_step_{args.step}"
    outpath.mkdir()
    tree_results = pd.read_csv(results_file, index_col=0)

    intermediate_meta = list()
    unfinished_with_distance = tree_results[(tree_results.dist >= args.distance) & (tree_results.finished == 0) & (tree_results.equal == 1)]
    with open(outpath/"test.src", "w") as s_out, open(outpath/"test.tgt", "w") as t_out:
        for i, row in enumerate(unfinished_with_distance.iterrows()):
            eq_data = dict()
            eq_data['id'] = i
            eq_data['orig_id'] = row[0]
            eq_data['dist'] = row[1].dist - 1
            eq_data['axiom'] = "ax_1"
            eq_data['subst_root'] = "+_0"
            eq = row[1].transformed_eq
            eq_data['paren'] = eq
            intermediate_meta.append(eq_data)

            if args.remove_paren:
                eq = eq.replace('(', '')
                eq = eq.replace(')', '')

            s_out.write(eq + "\n")
            t_out.write("ax_1 +_0\n")

    with open(outpath/'test.json', 'w') as f:
        json.dump(intermediate_meta, f)
