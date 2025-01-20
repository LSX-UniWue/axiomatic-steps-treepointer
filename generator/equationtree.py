from __future__ import annotations

import collections
import copy
import re
import uuid
from typing import Union, List

import sympy
from sympy import sympify
import treelib


class TreeData:
    def __init__(self, symbol, axioms=None):
        self.symbol = str(symbol)
        if not axioms:
            self.axioms = []
        else:
            self.axioms = axioms

    def __eq__(self, other: [str|TreeData]):
        if isinstance(other, str):
            return self.symbol == other
        else:
            return self.symbol == other.symbol

    def __str__(self):
        return self.symbol

    def __add__(self, other):
        return self.symbol + other

    def __repr__(self):
        return f'{self.symbol}, {self.axioms}'


class EquationTree(treelib.Tree):
    associativity = {'+': 0, '-': 0, '*': 0, '/': 0, '**': 1}
    pre = {'+': 0, '-': 0, '*': 1, '/': 1, '**': 2}
    func_dict = {"log": "ln"}
    op_dict = {"Add": '+', "Mul": '*', "Pow": "**", "Sub": '-', "Div": '/', '=': '='}  # , "Equality": '='}
    op_reversed = {v: k for k, v in op_dict.items()}

    def __init__(self, parsed_equation=None, str_equation=None, par_equation=None, tree=None, deep=False, identifier=None):
        if parsed_equation is not None:
            tree = EquationTree._build_from_sympy(parsed_equation)
        elif str_equation is not None:
            tree = EquationTree._build_from_str(str_equation)
        elif par_equation is not None:
            tree = EquationTree._build_from_parentheses(par_equation)

        super(EquationTree, self).__init__(tree=tree, deep=deep, identifier=identifier)

        self._node_counter = 0

    def create_node(self, tag=None, identifier=None, parent=None, data=None):
        data = TreeData(data)
        return super(EquationTree, self).create_node(tag, identifier, parent, data)

    def add_axiom_to_node(self, node_id: str, axiom: str):
        self.nodes[node_id].data.axioms.append(axiom)
    @staticmethod
    def get_tag(v_str: str):
        v_str = re.sub(r'_\d+', '', v_str)
        if v_str in EquationTree.op_reversed.keys():
            return EquationTree.op_reversed[v_str]
        elif v_str in ['x', 'y', 'z']:
            return "Symbol"
        elif v_str.isnumeric() or (len(v_str) >= 2 and v_str[0] == '-' and v_str[1:].isnumeric()) or v_str in ('E', 'pi'):
            return "Constant"
        elif v_str.isalpha():
            return "Function"
        else:
            raise RuntimeError("Invalid value")

    @staticmethod
    def _build_from_sympy(eq_nodes):
        func = str(eq_nodes.func).rstrip(">.\'").split('.')[-1]
        if func in {"One", "Zero", "NegativeOne", "Exp1", "Integer"}:
            func = "Constant"
        elif func in {"log", "sin", "cos", "tan"}:
            func = "Function"
        elif func == "exp":
            eq_nodes = sympify(f"E ** ({str(eq_nodes.exp)})", evaluate=False)
            return EquationTree._build_from_sympy(eq_nodes)
        elif func in ("Rational", "Half"):
            eq_nodes = sympy.sympify(str(eq_nodes), evaluate=False)
            return EquationTree._build_from_sympy(eq_nodes)
        if not (func in EquationTree.op_dict.keys() or func in {"Constant", "Symbol", "Function"}):
            raise TypeError("Unknown node type in tree: ", func)

        tree = EquationTree()
        if not eq_nodes.args:  # leaf
            tree.create_node(tag=func, data=eq_nodes)

        elif len(eq_nodes.args) == 1:
            f = str(eq_nodes.func)
            tree.create_node(tag=func, data=EquationTree.func_dict.get(f, f))
            branch = EquationTree._build_from_sympy(eq_nodes.args[0])
            tree.paste(tree.root, branch)

        else:
            branch_stack = list()
            for eq in reversed(eq_nodes.args):
                branch = EquationTree._build_from_sympy(eq)
                branch_stack.append(branch)
                if len(branch_stack) == 2:
                    sub_tree = EquationTree()
                    sub_tree.create_node(tag=func)
                    sub_tree.paste(sub_tree.root, branch_stack.pop())
                    sub_tree.paste(sub_tree.root, branch_stack.pop())
                    branch_stack.append(sub_tree)

            tree = branch_stack.pop()

        return tree

    @staticmethod
    def _build_from_str(exp: str):
        tree_item = collections.namedtuple("tree_item", ["value", "uuid"])

        def is_greater_precedence(op1, op2):
            try:
                return EquationTree.pre[op1] >= EquationTree.pre[op2]
            except KeyError:
                return True

        def build_from_stack(unary=False):
            tree = EquationTree()
            popped_operator = operator_stack.pop()
            tree.create_node(data=popped_operator.value, identifier=popped_operator.uuid, tag=EquationTree.get_tag(popped_operator.value))
            t1 = tree_stack.pop()

            if not unary and tree_stack:
                t2 = tree_stack.pop()

                if isinstance(t2.value, treelib.Tree):
                    tree.paste(tree.root, t2.value)  # TODO: check order in tree
                else:
                    tree.create_node(data=t2.value, identifier=t2.uuid, parent=popped_operator.uuid, tag=EquationTree.get_tag(t2.value))

            if isinstance(t1.value, treelib.Tree):
                tree.paste(tree.root, t1.value)
            else:
                tree.create_node(data=t1.value, identifier=t1.uuid, parent=popped_operator.uuid, tag=EquationTree.get_tag(t1.value))

            tree_stack.append(tree_item(tree, popped_operator.uuid))

        exp_list = exp.split()

        if len(exp_list) == 1:
            tree = EquationTree()
            tree.create_node(data=exp_list[0], tag=EquationTree.get_tag(exp_list[0]))
            return tree

        operator_stack = []
        tree_stack = []
        for i in exp_list:
            if i.isnumeric() or i in ['x', 'y', 'z', 'E', 'pi'] or (len(i) >= 2 and i[0] == '-' and i[1:].isnumeric()):
                tree_stack.append(tree_item(i, str(uuid.uuid1())))

            elif i.isalpha():
                operator_stack.append(tree_item(i, str(uuid.uuid1())))

            elif i in EquationTree.pre.keys():
                if not operator_stack or operator_stack[-1] == '(':
                    operator_stack.append(tree_item(i, str(uuid.uuid1())))

                elif is_greater_precedence(i, operator_stack[-1].value) and EquationTree.associativity[i] == 1:
                    operator_stack.append(tree_item(i, str(uuid.uuid1())))

                else:
                    while operator_stack and is_greater_precedence(operator_stack[-1].value, i) and EquationTree.associativity[i] == 0:
                        build_from_stack()
                    operator_stack.append(tree_item(i, str(uuid.uuid1())))

            elif i == '(':
                operator_stack.append('(')

            elif i == ')':
                while operator_stack[-1] != '(':
                    build_from_stack()
                operator_stack.pop()
                if operator_stack and isinstance(operator_stack[-1], tree_item) and EquationTree.get_tag(operator_stack[-1].value) == "Function":
                    build_from_stack(unary=True)

        while operator_stack:
            build_from_stack()

        t = tree_stack.pop()

        return t.value

    @staticmethod
    def _build_from_parentheses(par_str: str):
        def parse(tokens: collections.deque):
            tree = EquationTree()
            data = tokens.popleft().rstrip(') ')
            try:
                tag, data = data.split()
            except ValueError:
                tag = EquationTree.get_tag(data)

            # if tag not in ("Function", "CONSTANT", "SYMBOL"):
            #     tmp = tag
            #     tag = data
            #     data = tmp

            tag = tag.title()
            # data = data.rstrip(')')
            root = tree.create_node(tag=tag, data=data, identifier=str(uuid.uuid1()))

            if tag in ("Symbol", "Constant"):
                return tree

            if tag != "Function":
                lchild = parse(tokens)
                tree.paste(root.identifier, lchild)
            rchild = parse(tokens)
            tree.paste(root.identifier, rchild)

            return tree

        par_str = par_str.strip('()')
        splitted = collections.deque(par_str.split('('))
        return parse(splitted)

    def _clone(self, identifier=None, with_tree=False, deep=False):
        return EquationTree(tree=self if with_tree else None, deep=deep, identifier=identifier)

    @staticmethod
    def _switch_children(t: EquationTree, root: str):
        children = t.children(root)
        moved_child = t.remove_subtree(children[0].identifier)
        t.paste(root, moved_child)

    @property
    def variables(self):
        var_ids = {'x': 0, 'y': 1, 'z': 2}
        v = {var_ids[x.data] for x in self.filter_nodes(lambda x: x.tag == "Symbol")}
        return {f"var_{i}": i for i in range(len(v))}

    @property
    def variables_both_sides(self):
        lhs = self.subtree(self.children(self.root)[0].identifier)
        rhs = self.subtree(self.children(self.root)[1].identifier)

        return lhs.variables and rhs.variables

    @property
    def variable_names(self):
        v = {x.data for x in self.filter_nodes(lambda x: x.tag == "Symbol")}
        return list(v)

    def has_variable(self, var_name):
        lhs = self.subtree(self.children(self.root)[0].identifier)
        rhs = self.subtree(self.children(self.root)[1].identifier)
        out = list()
        if any([x.data == var_name for x in lhs.nodes.values()]):
            out.append(0)
        if any([x.data == var_name for x in rhs.nodes.values()]):
            out.append(1)

        return out

    def rename_nodes(self):
        for n in self.all_nodes():
            self.update_node(n.identifier, identifier=str(uuid.uuid1()))

    def enumerate_occurrences(self):
        oc_counter = collections.defaultdict(int)
        for n in self.all_nodes():
            data = str(n.data).split('_')[0]
            axioms = n.data.axioms
            occurrence = oc_counter[data]
            oc_counter[data] += 1
            self.update_node(n.identifier, data=TreeData(f"{data}_{occurrence}", axioms))

    def __str__(self):
        tree_inorder = self.inorder()
        if isinstance(tree_inorder, list):
            left, right = tree_inorder
            return f"{left} <SEP> {right}"
        else:
            return tree_inorder

    def contains_subtree(self, subTree):
        match_object = collections.namedtuple("match_object", ["matches", "match_dict"])

        def check_identical(tree_a_node, tree_b_node, perform_substitution=True, old_subst=None):
            # Check if the data of both roots is same and data of
            # left and right subtrees are also same
            if not isinstance(tree_a_node, str):
                tree_a_node = tree_a_node.identifier
            if not isinstance(tree_b_node, str):
                tree_b_node = tree_b_node.identifier

            tree_a_node_successors = self.children(tree_a_node)
            tree_b_node_successors = subTree.children(tree_b_node) if perform_substitution else self.children(
                tree_b_node)
            tree_b = subTree if perform_substitution else self
            new_subst = copy.copy(old_subst) if old_subst else dict()

            if perform_substitution and subTree[tree_b_node].tag == "Symbol":
                symbol = str(subTree[tree_b_node].data)
                if symbol not in new_subst:
                    new_subst[symbol] = tree_a_node
                    return match_object(True, new_subst)
                elif check_identical(tree_a_node, new_subst[symbol], False).matches:
                    return match_object(True, new_subst)
                else:
                    return match_object(False, old_subst)

            if len(tree_a_node_successors) != len(
                    tree_b_node_successors):  # different number of children -> trees can't match
                return match_object(False, old_subst)

            for i in range(len(tree_a_node_successors)):
                matches, subs = check_identical(tree_a_node_successors[i], tree_b_node_successors[i],
                                                perform_substitution, new_subst)  # check if each child is identical
                new_subst.update(subs)  # accept substitutions found in children
                if not matches:
                    return match_object(False, old_subst)

            if self[tree_a_node].tag == tree_b[tree_b_node].tag and str(self[tree_a_node].data).split('_')[0] == str(tree_b[tree_b_node].data).split('_')[0]:  # check if the leaves have identical values
                return match_object(True, new_subst)
            else:
                return match_object(False, old_subst)

        def match_trees(tree_node, subtree_node):
            all_matches = list()
            if not isinstance(tree_node, str):
                tree_node = tree_node.identifier
            match = check_identical(tree_node, subtree_node)
            if match.matches:
                all_matches.append(match_object(tree_node, match.match_dict))

            try:
                m1 = match_trees(self.children(tree_node)[0], subtree_node)
                if m1:
                    all_matches.extend(m1)
                m2 = match_trees(self.children(tree_node)[1], subtree_node)
                if m2:
                    all_matches.extend(m2)
            except IndexError:
                pass

            return all_matches

        if subTree.inorder() == self.inorder():
            return []

        return match_trees(self.root, subTree.root)

    def transform(self, other_structure: EquationTree, mapping: dict, mount_root: str) -> str:
        for node in other_structure.expand_tree(other_structure.root,
                                                sorting=False):  # sort the mapping dict according to the order in other_structure
            n = other_structure[node]
            if n.tag == "Symbol" and str(n.data) in mapping:
                temp = mapping[str(n.data)]
                del mapping[str(n.data)]
                mapping[str(n.data)] = temp

        new_tree = copy.deepcopy(other_structure)

        new_tree.rename_nodes()

        change_nodes = collections.defaultdict(list)
        for k in mapping.keys():
            for node in new_tree.filter_nodes(
                    lambda x: x.tag == "Symbol" and x.data == k):  # find nodes in new_tree that should be replaced with mapping k
                change_nodes[k].append(node)

        # if not other_structure.variables:
        #     self.paste(mount_root, other_structure)
        #     self.link_past_node(mount_root)
        # elif mapping.keys() != change_nodes.keys():
        if mapping.keys() != change_nodes.keys() and other_structure.variables:
            raise RuntimeError("No transformation possible")

        for k, v in mapping.items():
            sub = copy.deepcopy(self.subtree(v))
            for to_replace in change_nodes[k]:
                sub.rename_nodes()

                try:
                    idx = new_tree.children(new_tree.parent(to_replace.identifier).identifier).index(to_replace)
                    parent = new_tree.parent(to_replace.identifier).identifier
                    new_tree.paste(parent, copy.deepcopy(sub))
                    new_tree.remove_node(to_replace.identifier)  # remove the old to_replace
                    if idx == 0:
                        EquationTree._switch_children(new_tree, parent)

                except AttributeError:
                    new_tree = sub

        if self.root == mount_root:
            super(EquationTree, self).__init__(tree=new_tree)
        else:
            paste_parent = self.parent(mount_root).identifier
            self.paste(paste_parent, new_tree)
            children = self.children(paste_parent)
            if children[0].identifier == mount_root:  # this was the left branch
                moved_tree = self.remove_subtree(children[1].identifier)
                self.paste(paste_parent, moved_tree)

            self.remove_node(mount_root)

        return new_tree.root

    def inorder(self, root=None):
        output = ""

        first = root is None

        if not root:
            root = self.root
        if not isinstance(root, str):
            root = root.identifier

        successors = self.children(root)

        if self[root].tag == "Equality":
            output1 = self.inorder(successors[0])
            output2 = self.inorder(successors[1])
            return [output1, output2]

        elif len(successors) == 2:
            if not first:
                output += ' ( '
            output += self.inorder(successors[0]) + ' '
            try:
                output += self[root].data + ' '
            except TypeError:
                output += self.op_dict[self[root].tag] + ' '
            output += self.inorder(successors[1]) + ' '
            if not first:
                output += ' ) '

        elif len(successors) == 1:
            # output += tag + ' ('
            output += self[root].data + ' ( '
            output += self.inorder(successors[0])
            output += ' ) '

        else:
            output += str(self[root].data) if not str(self[root].data).startswith('-') else f" ( {self[root].data} ) "

        return output

    def write_out(self, root=None, var_dict=None):
        if not root:
            root = self.root
            self._node_counter = 0
            variables = {x.data for x in self.filter_nodes(lambda x: x.tag == "Symbol")}
            var_dict = {v: f"var_{i}" for i, v in enumerate(sorted(variables))}

        if not isinstance(root, str):
            root = root.identifier

        funcStr = self[root].tag + ',' if self[root].tag != "Zero" else "Constant,"
        if self[root].tag == "Symbol":
            varStr = var_dict[self[root].data] + ','
        elif self[root].data is None:
            varStr = ','
        else:
            varStr = self[root].data + ','

        numStr = str(self._node_counter) + ','
        depthStr = str(self.depth() - self.depth(root)) + ','
        self._node_counter += 1

        children = self.children(root)
        if len(children) == 0:
            funcStr = funcStr + '#,#,'
            varStr = varStr + '#,#,'
            numStr = numStr + '#,#,'
            depthStr = depthStr + '#,#,'
        elif len(children) == 1:
            p = self.write_out(children[0], var_dict)
            funcStr = funcStr + p[0] + '#,'
            varStr = varStr + p[1] + '#,'
            numStr = numStr + p[2] + '#,'
            depthStr = depthStr + p[3] + '#,'
        elif len(children) == 2:
            p0 = self.write_out(children[0], var_dict)
            p1 = self.write_out(children[1], var_dict)
            funcStr = funcStr + p0[0] + p1[0]
            varStr = varStr + p0[1] + p1[1]
            numStr = numStr + p0[2] + p1[2]
            depthStr = depthStr + p0[3] + p1[3]

        return [funcStr, varStr, numStr, depthStr]

    def write_out_paren(self):
        def _traverse_tree(root, out):
            if not isinstance(root, str):
                root = root.identifier

            node = self[root]
            if node.tag in ("Symbol", "Digit", "Constant"):
                out.append(f"{node.tag.upper()} {node.data}")
            else:
                out.append(node.data)

            for c in self.children(root):
                out.append(' (')
                _traverse_tree(c, out)
                out.append(')')

        buffer = list()
        _traverse_tree(self.root, buffer)

        return f"({''.join([str(b) for b in buffer])})"

    def insert_top(self, op: str, val: str):
        t = EquationTree()
        op_node = t.create_node(tag=op, data=EquationTree.op_dict[op])
        val_tag = EquationTree.get_tag(val)
        t.create_node(tag=val_tag, data=val, parent=op_node.identifier)
        t.paste(op_node.identifier, copy.deepcopy(self))
        super(EquationTree, self).__init__(tree=t)

    def subs(self, vars: List[str], subs_terms: List[Union[str, EquationTree]]):
        vars_nodes = [list(self.filter_nodes(lambda x: x.data == v)) for v in vars]
        subs_trees = [EquationTree(str_equation=t) if isinstance(t, str) else t for t in subs_terms]

        for vs, s in zip(vars_nodes, subs_trees):
            for v in vs:
                try:
                    v_parent = self.parent(v.identifier).identifier
                    left_child = self.children(v_parent).index(v) == 0
                    self.remove_node(v.identifier)
                    s.rename_nodes()
                    self.paste(v_parent, copy.deepcopy(s))
                    if left_child:
                        EquationTree._switch_children(self, v_parent)
                except AttributeError:
                    super(EquationTree, self).__init__(tree=s)

   
