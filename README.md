# axiomatic-steps-treepointer
This is the repo for our TMLR paper "Identifying Axiomatic Mathematical Transformation Steps using Tree-Structured Pointer Networks".
In this Readme we explain how to generate data, train models and evaluate them.
For generating data and inference install the dependencies in `requirements.txt`.

## Generate Data
The code of the dataset generator can be found in the folder `generator`. 
Run the code as follows: `python3 eqGen.py $n_equiv $n_trans $axioms_path` where `$n_equiv` is the number of start equations, `$n_trans` is the number of axiomatic transformation steps and `$out_path` is the folder where the generated data should be saved, e.g. `python3 eqGen.py 500 5000 /tmp/data`. 
You have to copy the file `axioms.csv` (or the file which specifies your own axioms) to `$out_path` before running the code.
We generally set `n_eqiv=10000, n_trans=10*n_equiv`.