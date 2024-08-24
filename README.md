SMGAN: Can be used to train a GAN to generate synthetic molecules, some of which are valid and unique molecules. See SMGAN.pdf and notebooks for details.

Quick start:
============================================================================================================================================================================================
- pip install -r requirements.txt

- start jupyter lab and open train.ipynb to train, and generate novel molecules.


Location: ./
===========================================================================================================================================================================================

- SMGAN.pdf/SMGAN.docx Contains a high-level overview of the problem, the training, hyperparameter tuning, and inference tuning experiments, as well, as possibleimprovements and future work.

- requirements.txt - Environment package list.

Location: ./data
===========================================================================================================================================================================================
Zinc_all_smiles_data.txt - contains the full 250K SMILES dataset, including the 10% used for training. 

Location: ./results
===========================================================================================================================================================================================
This folder contains results from training, hyperparameter tuning, and inference tuning.

Location: ./src:
===========================================================================================================================================================================================
- hyperparameter_tuning.ipynb - This notebook is explained in much more detail in SMGAN.pdf, but the basic highlight is that it is used to run 300 hyperparameter tuning experiments using RayTune. t the end of training, the models viability and quality metrics are calculated. Molecules are produced for qualitative inspection at the very end of this notebook.
(TODO: Update metrics calculations at the end of this notebook)

- generate_molecules.ipynb - This is more skeleton code, and the metrics also need to be updated to represent the latest calculations.
(TODO: Update metrics calculations at the end of this notebook)

- train.ipynb - In this notebook, the best found hyperparameters were used to train SMGAN for 1, 5, and 500 epochs. At the end of training, the models viability and 
quality metrics are calculated. Molecules are produced for qualitative inspection at the very end of this notebook.

- generation_tuner.ipynb - This notebook takes the model trained with the best found hyperparameters, and performs a search for the best max_len to use during inference. It also calculates the viability and quality metrics of the best inference parameters. 

- external_utils - This folder includes RDKit files that were manually downloaded from https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py This code is used to calculate synthesizeability

- net.py - contains the SMGAN network. 

- custom_dataset.py - is the custom SMILES dataset object.

- utils.py - contains helper and utility functions.