# AlertnessModels
This repository contains Python models for circadian and alertness estimation. All codes in `/python` folder. 

### Data Generation
`data_generation.py`
* Input: None
* Output: data in `/data/data.csv`, plots in `/plots`
`split_data.py`
* Input: data, default `/data/data.csv`
* Output: splitted daily data in `/data/train.csv`

### SVM
`simple_svm.py`
* Input: splitted data, default `/data/train.csv`
* Output: test reault, average linear regression error

### Sampling Model
`sampling_estimation.py`
* Inputs:
  * Data: `/data/train.csv`
  * Parameters(optional): `parameters`
* Output:
  * Plots: `/plots`

### Parameter Files
`parameters`: frozen parameters of sampling model
`standard`: standard parameters from mathematical model
