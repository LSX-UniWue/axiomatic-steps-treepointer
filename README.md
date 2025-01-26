# axiomatic-steps-treepointer
This is the repo for our TMLR paper "Identifying Axiomatic Mathematical Transformation Steps using Tree-Structured Pointer Networks" (https://openreview.net/forum?id=gLQ801ewwp).
In this Readme we explain how to generate data, train models and evaluate them.
For generating data and inference install the dependencies in `requirements.txt`.

## Generate Data
The code of the dataset generator can be found in the folder `generator`. 
Run the code as follows: `python3 eqGen.py $n_equiv $n_trans $axioms_path` where `$n_equiv` is the number of start equations, `$n_trans` is the number of axiomatic transformation steps and `$out_path` is the folder where the generated data should be saved, e.g. `python3 eqGen.py 500 5000 /tmp/data`. 
We generally set `n_eqiv=10000, n_trans=10*n_equiv`.
You have to copy the file `axioms.csv` (or the file which specifies your own axioms) to `$out_path` before running the code.

It is advisable to run multiple instances of the generator in parallel and merge the data afterwards.

## Training of Models
### TreePointerNet
The usage of TreePointerNet is described here: https://github.com/sj-w/tree_pointer_net

### Baselines
The baseline models are just regular fairseq models. See https://github.com/facebookresearch/fairseq on how to use this toolkit.

## Evaluation of Models
The code for evaluation is located in `inference`. Run the script `run_eval_steps_cluster.sh $checkpoints_folder $data $mode` where `$checkpoints_folder` is the folder to the fairseq checkpoints of the trained models, `$data` is the folder containing your test data and `$mode` is either `"tree"` (for TreePointerNet) or  `"sequence"` (e.g. transformer).

## Citation
If you find our code helpful, please cite the following paper:
```
Coming soon.
```