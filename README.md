# Modeling and active constrained optimization of C5/C6 isomerization via Artificial Neural Networks

This repository contains the code, datasets, and results published in our [paper](https://www.sciencedirect.com/science/article/pii/S0263876222001770) that was published in the Chemical Engineering Research and Design journal.

_Note on Data_: While we wait for UniSim approval, we have encrypted the dataset files `r490.csv` and `datasets/*`. 

## Scripts

- `create_dataset.py`: compiles raw UniSim result Excel input sheets into one comma delimited file, such as `r490.csv`.
- `split_train_test_repeated.py`: repeatedly splits the dataset into 10 random splits that are sampled via Latin Hypercube Sampling. Generates a split file that indicates which samples belong to the training or testing sets, such as `r490_splits.npy`.
- `evaluate_model.py`: trains and evaluates a given model configuration on a given dataset, on all given splits of the dataset.
- `nn.py`: artificial neural network model (ANN)
- `null_model.py`: a model that just predicts average proportions on the training set.
- `rbfm.py`: radial basis function (RBF) model
- `rm.py`: basis function linear regression model that can be configured as a linear regression model or, as shown in the paper, a quadratic nonlinear regression model.
- `opt_mp.py`: runs active experimentation simulations on a given dataset, using the given experimental configuration (e.g., noiseless, noisy, restricted transitions) and computational model/acquisition function configuration. 
- `pipeline_evaluation.py`: evaluates all models, as reported in the paper.
- `pipeline_run_active_exp.py`: runs all active experiment simulations, as reported in the paper.
- `fit_models_to_traces.py`: for each active experiment trace in the given trace file (one of the files in `traces/r490`), this fits an ANN model to all the input/output pairs collected during that active experiment and then uses the trained model to make predictions over all 10,000 combinations of reactor temperatures and pressures.

## Directories

- `cfgs/models/*.json`: configurations of models evaluated in Figure 5.
- `cfgs/*.json`: configurations of acquisition functions evaluated in Figure 10.
- `datasets/r490`: raw UniSim excel sheets containing the outputs of 10,000 simulations covering different combinations of reactor temperatures and pressures.
- `exp_cfgs`: active experiment configurations in Figure 10.
- `r490_results`: results of model evaluation in Figure 5.
- `traces/r490`: active experiment results (traces) for all combinations of experiments and acqusition functions.

## Analysis Notebooks

- `analysis_r1_temp.ipynb`: plots Figures 2 and 3.
- `analysis_product_correlations.ipynb`: plots Figure 4.
- `analysis_ann_results.ipynb`: plots Figure 5.
- `analysis_temp_vs_kpi_heatmaps.ipynb`: plots Figure 6.
- `analysis_constraint_ch_opt.ipynb`: plots Figure 7.
- `analysis_nn_uncertainty.ipynb`: plots Figure 9.
- `analysis_active_exp_traces.ipynb`: plots Figure 10.

- `analysis_input_kpi_correlation.ipynb`: generates Table 1.
- `analysis_global_optima.ipynb`: generates Table 2.
- `analysis_compare_surrogate_model_to_actual.ipynb`: generates Table 3.

## Requirements
The code has been tested on:
- Python 3.8.10
- tensorflow 2.8.0
- scikit-optimize 0.9.0
- pandas 1.3.4
- numpy 1.22.3
- sklearn 1.0.1
- xmltodict 0.12.0
- PIL 8.4.0
- matplotlib 3.4.3
- pingouin 0.5.1
- scipy 1.7.1
- seaborn 0.11.2
