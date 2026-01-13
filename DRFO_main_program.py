# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluate_model_performance_and_naive_fairness_fast_rmse, evaluation_gender, evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse

import os
import copy
import math
import heapq # for retrieval topK
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split 
import torch.nn.functional as F

from collaborative_models import matrixFactorization, sst_pred

from tqdm import tqdm
from fairness_training_for_drfo import model_dro, pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial_rmse_thresh_drfo

parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument("--data_name",type=str,default="Lastfm-360K",help="The name of the dataset eg. ml-1m, tenrec")
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='6',
                        help="device id to run")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--num_epochs", type=int, default= 500, help= "the max epoch of training")
parser.add_argument("--learning_rate", type= float, default= 1e-3, help="the learning rate for MF model")
parser.add_argument("--batch_size", type= int, default= 32768, help= "the batchsize for training")
parser.add_argument("--evaluation_epoch", type= int, default= 3, help= "the evaluation epoch")
parser.add_argument("--weight_decay", type= float, default= 1e-7, help= "the weight_decay for training")
parser.add_argument("--top_K", type=int, default= 5, help="the NDCG evaluation @ K")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
parser.add_argument("--saving_path", type=str, default= "./debug_drfo_thresh_eval/", help= "the saving path for model")
parser.add_argument("--result_csv", type=str, default="./debug_drfo_thresh_eval/result.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/", help= "the data path")
parser.add_argument("--fair_reg", type=float, default= 10 ,help= "the regulator for fairness")
parser.add_argument("--partial_ratio_male", type=float, default= 0.5, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.1, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--orig_unfair_model", type=str, default= "./pretrained_model/")
parser.add_argument("--gender_train_epoch", type=int, default= 1000, help="the epoch for gender classifier training")
parser.add_argument("--workspace_path", type=str, default= "./DRO/workspace", help="DROS data space")
parser.add_argument("--early_stop", type=int, default= 100, help="param for early stopping")
parser.add_argument("--DRFO_specific_lr", type=float, default = 1e-2, help="DRFO specific learning rate")
args = parser.parse_args()

print(args)

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=0):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)
#device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
# set hyperparameters
data_name = args.data_name 
saving_path = args.saving_path
emb_size = args.embed_size
output_size = args.output_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
evaluation_epoch = args.evaluation_epoch
weight_decay = args.weight_decay
fair_reg = args.fair_reg
partial_ratio_male = args.partial_ratio_male
partial_ratio_female = args.partial_ratio_female 
# random_samples = 100
top_K = args.top_K

data_path = os.path.join(args.data_path, args.data_name)

orig_sensitive_attr = pd.read_csv(data_path + "/sensitive_attribute.csv")
sensitive_attr = pd.read_csv(data_path + "/sensitive_attribute_random.csv")



pred_sensitive_attr = pd.read_csv(f"{args.workspace_path}/sst_pred/{data_name}/{data_name}_maleratio_{partial_ratio_male}_femaleratio_{partial_ratio_female}_seed_{args.seed}_gender_train_epoch{args.gender_train_epoch}.csv")

orig_model = torch.load(os.path.join(args.orig_unfair_model, args.data_name, "MF_orig_model"), map_location = torch.device("cpu"))



DRO_model = model_dro(data_name=args.data_name,
                      init_lambdas = fair_reg,
                      male_ratio=partial_ratio_male,
                      female_ratio=partial_ratio_female,
                      gender_train_epoch=args.gender_train_epoch,
                      orig_unfair_model=orig_model,
                      device=device,
                      seed = args.seed,
                      dataset_path = args.data_path,
                      workspace_path = args.workspace_path)

train_data = DRO_model.train_data 
valid_data = DRO_model.valid_data 
test_data = DRO_model.test_data 
num_uniqueUsers = max(train_data.user_id) + 1
# num_uniqueLikes = len(train_data.like_id.unique())
num_uniqueLikes = max(train_data.item_id) + 1
print("gammas:",DRO_model.gammas)


other_params = []
other_param_name = []
no_used = []
for name,param in DRO_model.named_parameters():
        if 'user_embedding_dict' not in name and 'lambda' not in name:
            other_params.append(param)
            other_param_name.append(name)
        else:
            no_used.append(name)


optim1 = optim.Adam(params=DRO_model.user_embedding_dict_dro.parameters(),lr=args.DRFO_specific_lr) #,weight_decay=weight_decay) 1e-2 5e-3 1e-3
optim2 = optim.Adam(params=other_params,lr=learning_rate,weight_decay=weight_decay)

print("optim2")
print(other_param_name)
import time 
begin_opt_time = time.time()

# rmse_thresh
if args.data_name == "Lastfm-360K":
    rmse_thresh = 0.327087092 / 0.98
elif args.data_name == "ml-1m":
    rmse_thresh = 0.412740352 / 0.98
else:
    raise ValueError("Not rmse thresh")


best_val_rmse, test_rmse_in_that_epoch, unfairness_val, unfairness_test, best_epoch, best_model = \
        pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial_rmse_thresh_drfo(DRO_model,
                                                                             optim1,
                                                                             optim2,
                                                                             train_data,
                                                                             num_epochs,
                                                                             learning_rate, 
                                                                             weight_decay, 
                                                                             batch_size, 
                                                                             valid_data, 
                                                                             test_data, 
                                                                             pred_sensitive_attr, 
                                                                             orig_sensitive_attr, 
                                                                             top_K, 
                                                                             fair_reg ,
                                                                             partial_ratio_male,
                                                                             partial_ratio_female, 
                                                                             device, 
                                                                             evaluation_epoch = evaluation_epoch, 
                                                                             unsqueeze = True,
                                                                             rmse_thresh = rmse_thresh)
os.makedirs(args.saving_path, exist_ok= True)
torch.save(DRO_model.state_dict(), args.saving_path + "/DRO_model")
torch.save(best_model.state_dict(), args.saving_path + "/best_model")

csv_folder = ''
for path in args.result_csv.split("/")[:-1]:
    csv_folder = os.path.join(csv_folder, path)

os.makedirs(csv_folder, exist_ok= True)

try:
    pd.read_csv(args.result_csv)
except:
    with open(args.result_csv,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["args", "best_val_rmse", "test_rmse_in_that_epoch", "unfairness_val_partial", "unfairness_test", "best_epoch"])

with open(args.result_csv,"a") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow([args, best_val_rmse, test_rmse_in_that_epoch, unfairness_val, unfairness_test, best_epoch])

