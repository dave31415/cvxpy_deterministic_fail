# cvxpy_deterministic_fail
Failing test case for CVXPY determinism

This is a simplified version of our trend filtering code. When you give the same input,
you can get non-deterministic outputs.

Just run it with

python -m pytest determ.py

or 

python determ.py


Tested on Python3 and versions

numpy '1.15.3'

scipy '1.1.0'

cvxpy '1.0.6'

