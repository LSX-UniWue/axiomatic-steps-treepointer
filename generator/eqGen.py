from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from sys import argv
import copy
import itertools
import json
import csv
import logging
import os.path
import random
from typing import List
import numpy as np
import tqdm

import sympy

from equationtree import EquationTree

# random.seed(0)
# np.random.seed(0)
# x, y, z, t = sympy.symbols('x y z', positive=True, real=True)

values = frozenset({'x', 'y', 'z', '0', '1', '2', '3', '4', '-1', '-2', '-3', '-4'})
subst_values = ['x', 'y', 'z']*10 + ['0', '1', '2', '3', '4', '-1', '-2', '-3', '-4']
constants = ['0', '1', '2', '3', '4', '-1', '-2', '-3', '-4']

variables = {f"var_{i}": i for i in range(3)}


@dataclass
class Equation:
    lhs: EquationTree
    rhs: EquationTree
    distance: int = 0
    rule: str = ""
    subst: str = ""  # root of the subtree which was substituted in the original tree
    subst_root: str = ""  # root of the tree that was inserted as subst
    previous_rhs: list = field(default_factory=list)
    label: int = 0

    def get_random_side(self):
        i = random.randint(0, 1)
        if i == 0:
            return self.lhs
        else:
            return self.rhs


def write_to_file(path: str, equations: List[Equation], maxDepth: int, mapping=None):
    data = [[] for _ in range(maxDepth + 1)]
    pickle_dict = dict()
    for i, eq in enumerate(equations):
        eq.rhs.enumerate_occurrences()
        tree = generate_pair_trees((eq.lhs, eq.rhs), mapping)
        depth = tree.depth()
        eq_str = str(tree)
        eq_str_paren = tree.write_out_paren()
        d = {'axiom': eq.rule, 'str': eq_str, 'paren': eq_str_paren, 'subst_id': eq.subst, 'subst_root_id': eq.subst_root,
             'dist': eq.distance, 'subst_root': str(eq.rhs.get_node(eq.subst_root).data), 'previous_rhs': eq.previous_rhs}
        if i == 0 and tree.depth() > maxDepth:
            data[0].append(d)
        else:
            data[depth].append(d)

        pickle_dict[eq_str] = eq

    n_samples = [len(data[i]) for i in range(len(data))]
    print(f"{str(path)}: {n_samples}, Σ {sum(n_samples)}")

    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def generate_equivalent_transformations(n_examples: int, allowed_depth: range, axioms: str,
                                        proto_examples: List[Equation], distance_step=0, no_progress=True,
                                        drop_invalid=False, upsample_leaves=False):
    def substitute_by_match(sample_id):
        if not distance_step:
            sample_id = random.randrange(len(generated_example_pairs))
            proto_example = copy.deepcopy(generated_example_pairs[sample_id].get_random_side())
        else:
            proto_example = copy.deepcopy(generated_example_pairs[sample_id].rhs)
        # print("Old: ", proto_example)
        candidates = map(lambda x: (proto_example.contains_subtree(x[0]), x[1]), axioms_dict.items())
        candidates = filter(lambda x: len(x[0]) > 0 and x[1].inorder() != proto_example.inorder(), candidates)
        candidates = list(candidates)
        if upsample_leaves:
            candidates = [c for c in candidates if any([not m.match_dict or proto_example[m.match_dict['x']].data in values for m in c[0]])]
            for c in candidates:
                c[0][:] = [m for m in c[0] if not m.match_dict or proto_example[m.match_dict['x']].data in values]
        try:
            while candidates:
                weights = [1 / used_axioms_count[f"{axioms_reversed[c[1]]}->{c[1]}"] for c in candidates]
                applied_candidate_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                applied_candidate = candidates.pop(applied_candidate_idx)
                while applied_candidate[0]:
                    applied_transformation_idx = random.choice(range(len(applied_candidate[0])))
                    applied_transformation = applied_candidate[0].pop(applied_transformation_idx)
                    ax = applied_candidate[1]
                    if distance_step and str(ax) in proto_example[applied_transformation.matches].data.axioms:
                        continue

                    new_example = copy.deepcopy(proto_example)
                    new_example_root = new_example.transform(ax, applied_transformation.match_dict, applied_transformation.matches)

                    # if distance_step:
                    new_example.add_axiom_to_node(new_example_root, axiom=str(axioms_reversed[ax]))

                    sn = f"{generated_example_pairs[sample_id].lhs}={new_example}" if distance_step else f"{proto_example}={new_example}"
                    if new_example.depth() in allowed_depth and sn not in all_examples and not check_invalid(
                            str(new_example)):
                        if distance_step:
                            if str(new_example) in generated_example_pairs[sample_id].previous_rhs or is_identical(generated_example_pairs[sample_id].lhs, new_example):
                                continue

                            generated_example_pairs.append(Equation(generated_example_pairs[sample_id].lhs, new_example, distance_step+1,
                                                                    f"{axioms_reversed[ax]}->{ax}",
                                                                    applied_transformation.matches, new_example_root,
                                                                    list(generated_example_pairs[sample_id].previous_rhs + [str(proto_example)])))
                        else:
                            generated_example_pairs.append(Equation(proto_example, new_example, 1, f"{axioms_reversed[ax]}->{ax}",
                                                                    applied_transformation.matches, new_example_root))

                        all_examples.add(sn)
                        generated_samples_str[sn] = new_example
                        used_axioms_count[f"{axioms_reversed[ax]}->{ax}"] += 1
                        candidates = None
                        break

        except IndexError:
            logging.info(f"No rule for {str(proto_example)} found")
            if drop_invalid:
                if sample_id not in invalid_idx:
                    invalid_idx.add(sample_id)
                else:
                    del generated_example_pairs[sample_id]
        except RuntimeError:
            logging.info(f"Invalid matchdict for {str(new_example)} and {str(applied_candidate[1])}")

    axioms_dict = dict()
    invalid_idx = set()
    generated_example_pairs = list(proto_examples)
    proto_len = len(proto_examples)
    generated_samples_str = dict()  # map str(equation) to its tree
    all_examples = set()
    init_size = len(generated_example_pairs)
    used_axioms_count = defaultdict(lambda: 1)

    for e in generated_example_pairs:
        all_examples.add(f"{e.lhs}={e.rhs}")
        # if not distance_step:
        #     all_examples.add(str(e.lhs))
        # all_examples.add(str(e.rhs))
    with open(axioms) as f:
        reader = list(csv.reader(f))
        for i, row in enumerate(reader[1:]):
            if not row or row[0].startswith('#') or row[0].startswith('\n'):
                continue

            lhs, rhs = row[0].split('=')
            lhs = EquationTree(str_equation=lhs)
            rhs = EquationTree(str_equation=rhs)
            axioms_dict[lhs] = rhs
            if not int(row[1]):
                axioms_dict[rhs] = lhs

            if not proto_examples:
                generated_example_pairs.append(Equation(copy.deepcopy(lhs), copy.deepcopy(rhs)))
                generated_samples_str[str(lhs)] = lhs
                generated_samples_str[str(rhs)] = rhs

    axioms_reversed = {v: k for k, v in axioms_dict.items()}
    if not no_progress:
        print("Generate Equivalent Transformations")
    with tqdm.tqdm(total=n_examples+init_size, disable=no_progress) as pbar:
        old_len = 0
        if distance_step:
            iterate_over_previous = iter(range(len(proto_examples)))
        else:
            iterate_over_previous = None
        while len(generated_example_pairs) < n_examples + init_size:
            if random.random() < 0.95 or proto_examples is not None:  # don't extend the sides in case of available prototypes (e.g. diffs)
                try:
                    substitute_by_match(next(iterate_over_previous) if iterate_over_previous is not None else None)
                except StopIteration:
                    break
            else:
                raise RuntimeError("Must not happen")

            new_len = len(generated_example_pairs)
            pbar.update(new_len-old_len)
            old_len = new_len

    return generated_example_pairs[proto_len:], generated_samples_str


def generate_pair_trees(eq_pair, mapping=None):
    def build_paired_tree(t1: EquationTree, t2: EquationTree):
        t = EquationTree()
        t.create_node(tag="Equality", data='=')
        t1 = copy.deepcopy(t1)
        t1.rename_nodes()
        t2 = copy.deepcopy(t2)
        t2.rename_nodes()
        t.paste(t.root, t1)
        t.paste(t.root, t2)

        return t

    pair_trees = [mapping[x] if isinstance(x, str) else x for x in eq_pair]
    pt = build_paired_tree(*pair_trees)

    return pt


def generate_equivalence_classes(n_examples: int, allowed_depth: range, axioms: str, weight_by_depth: bool = False):
    def free_symbols(eq: str):
        if 'x' in eq:
            return True
        elif 'y' in eq:
            return True
        elif 'z' in eq:
            return True
        return False

    def substitute_both_sides():
        with sympy.evaluate(False):
            #sample an axiom to use
            axiom = random.choice(available_axioms)

            #select substitute expressions
            if weight_by_depth: # (gewichtung nach länge der formel)
                x_subst, y_subst, z_subst = copy.deepcopy(random.choices(available_expressions, weights=expression_weights, k=3))
            else:
                x_subst, y_subst, z_subst = copy.deepcopy(random.choices(available_expressions, k=3))

            lhs, rhs = copy.deepcopy(axiom)
            lhs.subs(['x', 'y', 'z'], [x_subst, y_subst, z_subst])
            rhs.subs(['x', 'y', 'z'], [x_subst, y_subst, z_subst])
            if str(rhs) == str(lhs):
                return

            #check infinity
            if check_invalid(str(rhs)) or check_invalid(str(lhs)):
                print(rhs)
                print(lhs)
                return

            symbol_perm = ['x', 'y', 'z']
            random.shuffle(symbol_perm)
            lhs.subs(['x', 'y', 'z'], symbol_perm) #for some reason not working :(
            rhs.subs(['x', 'y', 'z'], symbol_perm) #not working for some reason (only works sometimes)

            # check depth of the generated example
            if lhs.depth() <= max(allowed_depth) and lhs.depth() <= max(allowed_depth):
                generated_example_pairs.append(Equation(copy.deepcopy(lhs), copy.deepcopy(rhs)))
                generated_samples_str[str(lhs)] = lhs
                generated_samples_str[str(rhs)] = rhs
                if lhs.depth() <= max(allowed_depth)-2:
                    available_expressions.append(copy.deepcopy(lhs))
                    expression_weights.append(1/(lhs.depth()+1))
                if rhs.depth() <= max(allowed_depth)-2:
                    available_expressions.append(copy.deepcopy(rhs))
                    expression_weights.append(1/(rhs.depth()+1))

    generated_example_pairs = list()
    generated_samples_str = dict()  # map str(equation) to its tree
    available_expressions = [EquationTree(str_equation=val) for val in subst_values]
    expression_weights = [1.0 / (expr.depth() + 1) for expr in available_expressions]
    available_axioms = []
    with open(axioms) as f:
        reader = list(csv.reader(f))
        for row in reader[1:]:
            if not row or row[0].startswith('#') or row[0].startswith('\n'):
                continue

            lhs, rhs = row[0].split('=')

            #add to example pairs
            lhs_tree = EquationTree(str_equation=lhs)
            rhs_tree = EquationTree(str_equation=rhs)
            generated_example_pairs.append(Equation(copy.deepcopy(lhs_tree), copy.deepcopy(rhs_tree)))
            generated_samples_str[str(lhs_tree)] = lhs_tree
            generated_samples_str[str(rhs_tree)] = rhs_tree

            if (free_symbols(lhs)) or (free_symbols(rhs)): #check if no free variables in axiom
                available_axioms.append((lhs_tree, rhs_tree))

    print("Generate Equivalence Classes")
    with tqdm.tqdm(total=n_examples) as pbar:
        old_len = 0
        while len(generated_example_pairs) < n_examples:
            substitute_both_sides()
            new_len = len(generated_example_pairs)
            pbar.update(new_len-old_len)
            old_len = new_len

    return generated_example_pairs, generated_samples_str


def numerically_equal(e1: str, e2: str) -> bool:
    e1 = sympy.sympify(e1)
    e2 = sympy.sympify(e2)

    e1_s = e1.free_symbols
    e2_s = e2.free_symbols
    n_free_symbols = max(len(e1_s), len(e2_s))

    if not (e1_s <= e2_s or e2_s <= e1_s):
        return False

    sampling_points = np.random.default_rng().uniform(1e-5, 2, (5, n_free_symbols))
    for v in sampling_points:
        e1_temp = copy.deepcopy(e1)
        e2_temp = copy.deepcopy(e2)
        for i, ve1 in enumerate(sorted(e1_s, key=lambda x: str(x))):
            e1_temp = e1_temp.subs(ve1, v[i])
        for i, ve2 in enumerate(sorted(e2_s, key=lambda x: str(x))):
            e2_temp = e2_temp.subs(ve2, v[i])

        r1 = e1_temp.evalf()
        r2 = e2_temp.evalf()
        if not np.isclose(float(r1), float(r2)):
            return False

    return True


def check_invalid(eq: str):
    e = sympy.sympify(eq)

    return e.has(sympy.zoo, sympy.nan)


def is_identical(lhs: EquationTree, rhs: EquationTree):
    for l, r in zip(lhs.expand_tree(sorting=False), rhs.expand_tree(sorting=False)):
        if lhs.get_node(l).data != rhs.get_node(r).data:
            return False

    return True


if __name__ == "__main__":
    # outpath = "../data/axioms_with_div"
    out_id = uuid.uuid4()
    outpath = argv[3]
    print(outpath)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    axioms_file = "axioms.csv"
    train_max_depth = 8
    test_max_depth = 14
    train_range = range(0, train_max_depth)
    test_range = range(train_max_depth, test_max_depth)
    start_samples, mapping = generate_equivalence_classes(int(argv[1]), range(0, test_max_depth), os.path.join(outpath, axioms_file), weight_by_depth=True)
    all_data = list()
    transformed_dist_1, mapping_subst = generate_equivalent_transformations(int(argv[2]), range(0, test_max_depth),
                                                                            os.path.join(outpath, axioms_file),
                                                                            start_samples, no_progress=False)
    mapping.update(mapping_subst)
    transformed_dist_2, mapping_subst = generate_equivalent_transformations(len(transformed_dist_1),
                                                                            range(0, test_max_depth),
                                                                            os.path.join(outpath, axioms_file),
                                                                            transformed_dist_1, distance_step=1,
                                                                            no_progress=False)
    mapping.update(mapping_subst)
    transformed_dist_3, mapping_subst = generate_equivalent_transformations(len(transformed_dist_2),
                                                                            range(0, test_max_depth),
                                                                            os.path.join(outpath, axioms_file),
                                                                            transformed_dist_2, distance_step=2,
                                                                            no_progress=False)
    mapping.update(mapping_subst)
    transformed_dist_4, mapping_subst = generate_equivalent_transformations(len(transformed_dist_3),
                                                                            range(0, test_max_depth),
                                                                            os.path.join(outpath, axioms_file),
                                                                            transformed_dist_3, distance_step=3,
                                                                            no_progress=False)
    mapping.update(mapping_subst)
    transformed_dist_5, mapping_subst = generate_equivalent_transformations(len(transformed_dist_4),
                                                                            range(0, test_max_depth),
                                                                            os.path.join(outpath, axioms_file),
                                                                            transformed_dist_4, distance_step=4,
                                                                            no_progress=False)
    mapping.update(mapping_subst)
    all_data.extend(transformed_dist_1)
    all_data.extend(transformed_dist_2)
    all_data.extend(transformed_dist_3)
    all_data.extend(transformed_dist_4)
    all_data.extend(transformed_dist_5)

    train_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(0, train_max_depth) else 0 for x in all_data]
    test_dist1_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(train_max_depth, test_max_depth) else 0 for x in transformed_dist_1]
    test_dist2_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(train_max_depth, test_max_depth) else 0 for x in transformed_dist_2]
    test_dist3_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(train_max_depth, test_max_depth) else 0 for x in transformed_dist_3]
    test_dist4_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(train_max_depth, test_max_depth) else 0 for x in transformed_dist_4]
    test_dist5_idx = [1 if max(x.lhs.depth(), x.rhs.depth())+1 in range(train_max_depth, test_max_depth) else 0 for x in transformed_dist_5]

    write_to_file(os.path.join(outpath, f"train_ec{argv[1]}_et{argv[2]}-{out_id}.json"),
                  list(itertools.compress(all_data, train_idx)), maxDepth=train_max_depth)
    write_to_file(os.path.join(outpath, f"test_ec{argv[1]}_et{argv[2]}_dist1-{out_id}.json"),
                  list(itertools.compress(transformed_dist_1, test_dist1_idx)), maxDepth=test_max_depth)
    write_to_file(os.path.join(outpath, f"test_ec{argv[1]}_et{argv[2]}_dist2-{out_id}.json"),
                  list(itertools.compress(transformed_dist_2, test_dist2_idx)), maxDepth=test_max_depth)
    write_to_file(os.path.join(outpath, f"test_ec{argv[1]}_et{argv[2]}_dist3-{out_id}.json"),
                  list(itertools.compress(transformed_dist_3, test_dist3_idx)), maxDepth=test_max_depth)
    write_to_file(os.path.join(outpath, f"test_ec{argv[1]}_et{argv[2]}_dist4-{out_id}.json"),
                  list(itertools.compress(transformed_dist_4, test_dist4_idx)), maxDepth=test_max_depth)
    write_to_file(os.path.join(outpath, f"test_ec{argv[1]}_et{argv[2]}_dist5-{out_id}.json"),
                  list(itertools.compress(transformed_dist_5, test_dist5_idx)), maxDepth=test_max_depth)
