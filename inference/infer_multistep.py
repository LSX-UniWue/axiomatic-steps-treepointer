import argparse
import copy
import dataclasses
import itertools
import json
import re
import multiprocessing
from mpmath import almosteq, isfinite, mpc, ln, e as E, pi, sin, cos, tan
from wrapt_timeout_decorator import timeout
import numpy as np

from data_generators.equationtree import EquationTree


@dataclasses.dataclass
class Prediction:
    id: int
    eq: str
    dist: int
    gold_ax: str
    gold_pos: str
    ax: list = dataclasses.field(default_factory=list)
    pos: list = dataclasses.field(default_factory=list)
    probs: list = dataclasses.field(default_factory=list)
    transformed_eq: list = dataclasses.field(default_factory=list)
    finished: list = dataclasses.field(default_factory=list)


def numerically_equal(e1: str, e2: str) -> int:
    @timeout(1)
    def _numerically_equal(e1, e2):
        regex = r"(-?\d+)"
        subst = "mpc(\\1.0)"

        free_symbols = ['x', 'y', 'z']
        n_free_symbols = len(free_symbols)

        sampling_points = np.random.default_rng().uniform(1e-5, 2, (10, n_free_symbols))
        e1 = re.sub(regex, subst, e1, 0, re.MULTILINE)
        e2 = re.sub(regex, subst, e2, 0, re.MULTILINE)
        for v in sampling_points:
            s1 = e1
            s2 = e2
            for i, ve1 in enumerate(free_symbols):
                s1 = s1.replace(ve1, f"mpc({v[i]})")
            for i, ve2 in enumerate(free_symbols):
                s2 = s2.replace(ve2, f"mpc({v[i]})")

            try:
                r1 = eval(s1)
                r2 = eval(s2)
                if not (isfinite(r1) and isfinite(r2)):
                    return -1
            except ZeroDivisionError:
                return -1

            if not almosteq(r1, r2):
                return 0

        return 1

    for _ in range(10):
        try:
            return _numerically_equal(e1, e2)
        except (TimeoutError, OverflowError, MemoryError, RecursionError):
            pass

    return -1


def is_identical(eq: str):
    eq = eq.replace("Mul", '*')   # fixme: fix this in the data generation process
    eq = eq.replace("Add", '+')
    eq = eq.replace("Sub", '-')
    eq = eq.replace("Div", '/')
    eq = eq.replace("Pow", '**')
    eq = re.sub(r'_\d+', '', eq)
    tree = EquationTree(par_equation=eq)
    l_child, r_child = tree.children(tree.root)
    for l, r in zip(tree.expand_tree(l_child.identifier, sorting=False), tree.expand_tree(r_child.identifier, sorting=False)):
        if tree.get_node(l).data != tree.get_node(r).data:
            return False

    return True


def apply_axiom(eq: str, axiom: str, root: str):
    eq = eq.replace("Mul", '*')   # fixme: fix this in the data generation process
    eq = eq.replace("Add", '+')
    eq = eq.replace("Sub", '-')
    eq = eq.replace("Div", '/')
    eq = eq.replace("Pow", '**')
    eq_clean = re.sub(r'_\d+', '', eq)
    axiom = [re.sub(r'^\s*\(\s*(-\s*\d)\s*\)\s*$', r'\1', s) for s in axiom.split('->')]
    axiom = [EquationTree(str_equation=s) for s in axiom]
    tree = EquationTree(par_equation=eq_clean)
    tree_clean = copy.deepcopy(tree)
    _, r_child = tree.children(tree.root)
    r_subtree = tree.subtree(r_child.identifier)
    res = r_subtree.contains_subtree(axiom[1])
    r_subtree.enumerate_occurrences()
    res = [m for m in res if tree.get_node(m.matches).data == root]
    if not res:
        return None

    r_subtree_clean = tree_clean.subtree(r_child.identifier)
    r_subtree_clean.transform(axiom[0], res[0].match_dict, res[0].matches)
    r_subtree_clean.enumerate_occurrences()
    tree.remove_subtree(r_child.identifier)
    tree.paste(tree.root, r_subtree_clean)

    return tree


def evaluate_predictions(pred: Prediction):
    evaluations = list()
    for b, (ax, pos) in enumerate(zip(pred.ax, pred.pos)):
        matches = 0
        data_step = 0
        finished = 0

        transformed_tree = apply_axiom(pred.eq, ax, pos)
        if ax == pred.gold_ax and pos == pred.gold_pos:
            data_step = 1
        if transformed_tree is not None:
            transformed_tree_paren = transformed_tree.write_out_paren()
            if is_identical(transformed_tree_paren):
                finished = 1
            matches = 1
            str_equation = str(transformed_tree)
            l_child, r_child = str_equation.split('=')
            r_child = re.sub(r'_\d+', '', r_child)
            equal = numerically_equal(l_child, r_child)
        else:
            transformed_tree_paren = ""
            equal = 0

        evaluations.append(f'{pred.id},{pred.gold_ax},{pred.gold_pos},{ax},{pos},{data_step},{b},{pred.dist},{pred.eq},{transformed_tree_paren},{finished},{matches},{equal}\n')

    return evaluations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument('metadata')
    parser.add_argument('outfile')
    parser.add_argument('--axioms')
    args = parser.parse_args()

    invalid_predictions = 0
    axioms_dict = dict()
    predictions = dict()
    with open(args.metadata) as f:
        data = json.load(f)
        if not args.axioms:
            for d in data:
                axioms_dict[d['label']] = d['axiom']

    if not args.axioms:
        with open('/tmp/axioms.json', 'w') as f:
            json.dump(axioms_dict, f)
    else:
        with open(args.axioms) as f:
            json_axioms_dict = json.load(f)
            axioms_dict = {int(k): v for k, v in json_axioms_dict.items()}

    with open(args.predictions) as f:
        model_output = f.readlines()
        for l in filter(lambda x: x.startswith("H-"), model_output):
            m = re.match(r'H-(\d+)\t.*\tax_(\d+) (.*_.*)', l)
            if m is None:
                invalid_predictions += 1
                continue
            pred_id = int(m.group(1))
            gold_data = data[pred_id]
            if pred_id not in predictions:
                pred = Prediction(pred_id, gold_data['paren'], gold_data['dist'], gold_data['axiom'], gold_data['subst_root'])
                predictions[pred_id] = pred

            predictions[pred_id].ax.append(axioms_dict[int(m.group(2))])
            predictions[pred_id].pos.append(m.group(3))
            predictions[pred_id].probs.append(m.group(1))

    with multiprocessing.Pool(processes=16) as pool:
        evaluations = pool.map(evaluate_predictions, list(predictions.values()))
    # for p in list(predictions.values()):
    #     evaluate_predictions(p)

    header = ("id", "data_axiom", "data_position", "pred_axiom", "pred_position", "data_step", "beam", "dist", "eq", "transformed_eq", "finished", "matches", "equal")
    with open(args.outfile, 'w') as out_file:
        out_file.write(','.join(header) + '\n')
        for ev in itertools.chain.from_iterable(evaluations):
            out_file.write(ev)

    print("Invalid: ", invalid_predictions)
