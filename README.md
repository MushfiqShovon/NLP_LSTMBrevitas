# Quantization  with Brevitas on a LSTM model for text classification  

# Discriminative Model

#### All the program can be run on terminal. for Discriminative model, go to the directory for discrimnative model with following command in terminal:
cd DiscriminativeClassifier  
#### In data directory, all the data files for ag_news dataset is saved. For experimenting with other dataset, the data files need to be placed in this directory.

## System Preparation (Prerequisite):
#### Run following bash command in terminal for system preparation. This will install all prerequisite:
sh Prerequisite.sh

## For running full precision model run following command in bash:
mkdir ModelParameter_FP  
python training_FP.py

## For running 4bit quantization run following command in bash:
mkdir ModelParameter_4bit  
python training_4bit.py


## For running 2bit quantization run following command in bash:
mkdir ModelParameter_2bit  
python training_2bit.py


## For running 1bit quantization run following command in bash:
mkdir ModelParameter_1bit  
python training_1bit.py
