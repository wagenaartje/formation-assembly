### Decentralized assembly of arbitrary formations using identical agents
#### AE4350 Assignment
by Thomas Wagenaar
<hr>

> This repository provides all the code for generating the results and the subsequent figures used in my report for the course _AE4350 Bio-inspired intelligence and learning for aerospace applications_. The size of this repository is fairly large (1.5+ GB), but this is simply because of the datafiles necessary to store the results. To save space, one can clone everything but the `/results` folder.
> 

<p align='center'><img src="https://github.com/wagenaartje/AE4350-formation/blob/cma/figures/a_heat_map.png?raw=true"></p>

### :fast_forward: How to run
First of all, the code was developed in Python 3.8.5 with the packages listed in `requirements.txt`. If you have Python 3.9+, you might have to change some imports of `pymoo` but everything should work. To start a single optimization run, only three files are necessary:

- [base_config.json](https://github.com/wagenaartje/AE4350-formation/blob/cma/base_config.json): contains all the hyperparameters
- [evaluation.py](https://github.com/wagenaartje/AE4350-formation/blob/cma/evaluation.py): provides the fitness evaluation functions
- [train.py](https://github.com/wagenaartje/AE4350-formation/blob/cma/train.py): implements CMA-ES for optimization

For a simple run, only these steps are necessary:

1. Set the hyperparameters in `base_config.json`
2. Run `train.py`

For the given settings in `base_config.json`, training should take approximately 90 minutes. After training has completed, a `/results` folder will have been created with the start timestamp. In the folder one can find three files:

- `config.json`: a copy of the hyperparameters for later use
- `fitnesses.dat`: the best fitness for each epoch
- `genomes.dat`: parameters of the genome with the best fitness for each epoch

The latter two files can be loaded with `np.fromfile()` (**not** `np.load()`).

<p>  </p>



### :floppy_disk: Folder explanation
Although the three files mentioned above are all that is necessary to perform the optimization procedure, some other scripts are present for batch training and result analysis. A small overview of the contents of each folder is given below:

- `/analysis`: contains the scripts that generated the figures for the report
- `/figures`: output directory of the above scripts, contains the figures
- `/results`: contains existing results from the batch training scripts
- `/train_scripts`: scripts for batch training, i.e. they perform multiple optimization runs with different settings

**Note!** If you want to run any scripts in this repository, it must be done from the top repository directory. Do not `cd` into directories to run the scripts.

<p>  </p>

### :mailbox: Contact
t.wagenaar-1@student.tudelft.nl
