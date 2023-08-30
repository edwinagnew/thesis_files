

This repository contains the code supplement for the MFoCS thesis "Quantum Polynomials in the ZXW calculus". 

Contents:
- `pyfile.py`, `spiders.py` contain core methods for building/evaluating ZXW diagrams. `polys.py` contains core methods for working with the polynomial representation of states.
- `basic_functionality.ipynb` gives some basic examples of this code in action.
- `d_spiders.py` generalises `spiders.py` to qudits. Qudit polynomial arithmetic is examined in `qudits.ipynb`. It is found that Z performs a more complicated multiplication than in the qubit case.
- `entanglement_algs.ipynb` benchmarks the polynomial-based separability test against a Schmidt-rank-based method. The results are also shown below:

<img src="https://github.com/edwinagnew/thesis_files/assets/42814611/340ff9f0-f5dd-437f-935f-a4cd22535d86" width="500">
