# TVAR
TVAR is a tissue-specific functional annotation tool of non-coding variants based on multi-label deep learning. The framework's input is the vcf profiles of non-coding variants. The output is the corresponding functional score vectors across the GTEx 49 tissues. TVAR is mainly divided into two components: Feature extraction and result analysis module running on the CPU (TVAR-CPU). 2. Model training and scoring module running on GPU (TVAR-GPU).

```{r}
# The input data file is the vcf file of non-coding variants. The following command is used to finish the feature extraction process: 
python TVar_cpu.py -m fea -i ./input/input.vcf
# The feature extraction program requires downloading a large number of genome-wide annotation files. For details, see the paper "TVAR: Assessing Tissue-specific Functional Effects of Non-coding Variants with Deep Learning".
```
```{r}
# The input data file is the features file (also with labels)of non-coding variants. The following command is used to finish the model training process:
python TVar_gpu.py -m train  -i ./input/input.gz
```
```{r}
# Train the model on the GTEx data set
python TVar_gpu.py -m cv  -i ./input/input.gz
# Train the model with five-fold cross-validation
```
```{r}
TVar's scoring module is used as follows: 
python TVar_gpu.py -m score -i ./input/input.vcf
# Generate the tissue-specific functional scores of the variants
```  
```{r}
The convolutional neural network (CNN) and long short-term memory (LSTM), can also be selected to replace the DNN model: 
python TVar_lstm_cnn.py -d CNN -m cv  -i ./input/input.gz
python TVar_lstm_cnn.py -d LSTM -m cv  -i ./input/input.gz
python TVar_lstm_cnn.py -d CNN -m score  -i ./input/input.gz
python TVar_lstm_cnn.py -d LSTM -m score  -i ./input/input.gz
```  

```{r}
The PCA method used in the feature extraction  can be replaced with autoencoder (AE) and embedding (EMB): 
python TVar_ae_embedding.py -d AE -m fea  -i ./input/input.gz
python TVar_ae_embedding.py -d EMB -m fea  -i ./input/input.gz
```  

We used TVAR to obtain the tissue-specific annotation scores of all the 176,673 variants from the GWAS catalog (version e105) and constituted the GWAS functional non-coding variants atlas (GFNCVA.score.gz and GFNCVA.score.gz.tbi).
```{r}
The tabix can be used to score the variants in the GFNCVA: 
tabix GFNCVA.score.gz 1:10000000-20000000
```  

TVAR was based on open-source Python 3.6 libraries. The deep learning network's implementation was based on Numpy 1.15.4, Scipy 1.1.0, Tensorlayer 1.11.1 (GPU version) and Tensorflow 1.11.0 (GPU version). After the testing, TVAR has been working correctly on Ubuntu Linux release 20.04. We used the NVIDIA Tesla T4 for model training and testing.
We provided a list of packages (tvar.yml) required by the TVAR runtime environment. The command: conda env create -f tvar.yml can complete the configuration of the TVAR operating environment.
