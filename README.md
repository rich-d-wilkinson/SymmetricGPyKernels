# Symmetric kernels for GPy

This code provides functionality to make a symmetric kernel from any of the stationary GPy kernels. To use it

- Install the latest version of [GPy](https://github.com/SheffieldML/GPy)
- Download symmetric.py and possibly __init__.py (not sure whether this is needed or not).
- Add `from symmetric import Symmetric` to the top of your python code.
- The `Symmetric(k, perm)` should create a symmetric version of the kernel `k` which obeys permuations given by `perm`.

See the examples for how to specify permuation groups. You can view the notebook for a symmetric 2d problem, or HigherD.py or HigherD2.py for more complicated higher dimensional problems.
