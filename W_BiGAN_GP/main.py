
from datetime import datetime
import torch
import os
import sys
from scripts import util
device = torch.device("cpu")
#%% varriables for training
istrain = True
batch_size =128
epoch = 7000
run_num = 1

path_model = "/Users/mohamedshanil/Desktop/iitb_ML_project/bigan-pep-test/W_BiGAN_GP/model_saved/"
if not os.path.exists(path_model):
    os.makedirs(path_model)
path_result = path_model + "/results_" + str(run_num)+  "_backup/"  ### analysis output dir
if not os.path.exists(path_result):
    os.makedirs(path_result)

## training data path
path_data = "/Users/mohamedshanil/Desktop/iitb_ML_project/bigan-pep-test/data_master/" # root for training data
fasta_AMPs = ["/Users/mohamedshanil/Desktop/iitb_ML_project/bigan-pep-test/data_master/amps/dbaasp/dbaasp_APR_processed.fasta"]
fasta_AMYs =["/Users/mohamedshanil/Desktop/iitb_ML_project/bigan-pep-test/data_master/amyloid/amyloid-combined.fasta"]
#%% other varriables for analysis
## for analysis.py
batch_generate =1000
## for analysis_generated_seqs.py
fasta_random = [path_data+ "random_pep_uni.fasta"]
fasta_uperin = [path_data+ "uperin.fasta"]
fasta_generated = [path_result+"seqs_generated" + str(batch_generate) +".fasta"]
## fasta_generated should be first in list in following list
fasta_all = fasta_generated + fasta_AMPs + fasta_AMYs + fasta_random + fasta_uperin
conditions_AMPs = {"AMP Score": 0.95}
score_AMYs = 0.95

#%%dataloader
path_PC6 = path_data + "physical_chemical_6.txt"
table = util.get_conversion_table(path_PC6)
seqs ={}
for f in fasta_AMPs + fasta_AMYs:
    cur_seq = util.read_fasta(f)
    print("seqs_num= {}".format(len(cur_seq)))
    seqs.update(cur_seq) ### merging two dict
print("Total seqs = {}".format(len(seqs)))

#### PC6 encoding
encoded = util.get_encoded_seqs(seqs, table)
dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#%% Training
"""
# Training
"""
from models_nn import train
if istrain == True:
    start_time = datetime.now().strftime("%H:%M:%S")

    G, loss_all, collected_seqs = train.training(path_model, epoch = epoch, train_data=dataloader, run_num = run_num)

    finish_time = datetime.now().strftime("%H:%M:%S")
    print("Current Time = {} \n Finish Time = {}".format(start_time, finish_time))
#%% model summary
# from models_nn import train
# train.models_summary(path_model)

# #%%analysis






