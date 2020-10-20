# Split-and-conquer for learning finite Gaussian Mixtures (SCGMM)

The folder contains the code for both the simulated and real data analysis in the paper entitled **distributed learning of finite Gaussian mixtures**. The code for these analysis are in two separate subdirectories named real_data and simulation respectively. We will go through these two folders one by one.


## Requirements
The code is written in Python 3.7.3. Package dependencies are listed in requirements.txt.




## Real Data
The real_data folder has the following structure
```
|____NIST___
|           |___nn_feature_extractor.py
|           |___README.txt
|
|____CAM____
            |___preprocessing.py
            |___fit_local_machine.py
            |___aggregation_gmr.py
            |___post_analysis_gmr.py
```

The NIST folder contains the code for the NN to extract the features of the NIST handwritten digits. The preprocessing and the use of the feature extractor can be found in README.txt. The code for the split-and-conquer experiment is very similar to what we have in the simulation study, we refer interested reader to go over simulation/demo.py for details.


The CAM folder contains four files that corresponding to the preprocessing, local inference, aggregation, and visualization respectively. After the dataset is downloaded, these files can be run from top to bottom to perform the analysis on the atmospheric dataset.


## Simulation
The simulation folder contains the following files or subdirectories:
```
|_____CTDGMR____
|               |___gmm_reduction.py
|               |___utils.py
|
|_____GMMpMLE.py
|
|_____demo.py
```

The gmm_reduction.py contains the python object used for aggregation by GMR.
The GMMpMLE.py is used to compute the pMLE of the finite Gaussian mixture.
The demo.py contains the code used to compute the global, GMR, Median, and KL-averaging estimators in the simulation study. The code can be easily modified to perform the experiment on the NIST dataset.

To run a simulation study, please run python demo.py and specify the true model parameters in line 317--329. You can specify the number of local machines in line 65.
