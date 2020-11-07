# TVAR
TVAR is a tissue-specific functional annotation tool of non-coding variants based on multi-label deep learning. The framework's input is the vcf profiles of non-coding variants. The output is the corresponding functional score vectors across the GTEx 49 tissues. TVAR is mainly divided into two components: Feature extraction and result analysis module running on the CPU (TVAR-CPU). 2. Model training and scoring module running on GPU (TVAR-GPU).
```{r}
# The input data file is the vcf file of non-coding variants. The following command is used to finish the feature extraction process: 
python TVar_cpu.py -m fea -i ./input/input.vcf
```
```{r}
# The input data file is the features file (also with labels)of non-coding variants. The following command is used to finish the model training process:
python TVar_gpu.py -m train  -i ./input/input.gz
# Train the model on the GTEx data set
python TVar_gpu.py -m cv  -i ./input/input.gz
# Train the model with five-fold cross-validation
```
```{r}
TVar's scoring module is used as follows: 
python TVar.py -m score -i ./input/input.vcf
# Generate the tissue-specific functional scores of the variants
```  
TVAR is based on open-source Python 3.7.4 libraries. The deep learning network's implementation was based on Numpy 1.16.1, Scipy 1.1.0, Tensorlayer 1.11.1 (GPU version) and Tensorflow 1.14.0 (GPU version). After the testing, TVAR has been working correctly on Ubuntu Linux release 18.04. We used the NVIDIA Tesla T4 for model training and testing.
