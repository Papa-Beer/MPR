
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluation_gender_new, evaluation_gender
from sklearn.model_selection import train_test_split 
import os
import copy
import math
import heapq # for retrieval topK
import random
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collaborative_models import matrixFactorization, sst_pred
from tqdm import tqdm
import json 


parser = argparse.ArgumentParser(description='fairRec')
parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='1',
                        help="device id to run")
parser.add_argument("--embed_size", type=int, default= 64, help= "the embedding size of MF")
parser.add_argument("--output_size", type=int, default= 1, help="the output size of MF")
parser.add_argument("--evaluation_epoch", type= int, default= 1, help= "the evaluation epoch")
parser.add_argument("--top_K", type=int, default= 5, help="the NDCG evaluation @ K")
parser.add_argument('--seed', type=int, default=1, help="the random seed")
# parser.add_argument("--saving_path", type=str, default= "./debug/", help= "the saving path for model")
# parser.add_argument("--result_csv", type=str, default="./debug/result.csv", help="the path for saving result")
parser.add_argument("--data_path", type=str, default="./datasets/", help= "the data path")
parser.add_argument("--partial_ratio_male", type=float, default= 0.5, help= "the known ratio for training sensitive attr male ")
parser.add_argument("--partial_ratio_female", type=float, default= 0.5, help= "the known ratio for training sensitive attr female ")
parser.add_argument("--orig_unfair_model", type=str, default= "/NAS/zhangjz/MPR_diff_seed_construct_batch/pretrained_model")
parser.add_argument("--gender_train_epoch", type=int, default= 1000, help="the epoch for gender classifier training")
parser.add_argument("--workspace",type=str,default="./DRO/workspace")
parser.add_argument("--dataset",type=str,default="Lastfm-360K",help="The dataset we are using")
parser.add_argument("--batchsize",type=int,default=128,help="The classifer batchsize")
args = parser.parse_args()

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=0):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
emb_size = args.embed_size
output_size = args.output_size
workspace = args.workspace


def process_data(data_name, male_know_size, female_know_size):

    if data_name == 'ml-1m':
        train_df = pd.read_csv("./datasets/ml-1m/train.csv")
        valid_df = pd.read_csv("./datasets/ml-1m/valid.csv")
        test_df = pd.read_csv("./datasets/ml-1m/test.csv")
        genders = pd.read_csv("./datasets/ml-1m/sensitive_attribute_random.csv")
    elif data_name == "Lastfm-360K":
        train_df = pd.read_csv("./datasets/Lastfm-360K/train.csv")
        valid_df = pd.read_csv("./datasets/Lastfm-360K/valid.csv")
        test_df = pd.read_csv("./datasets/Lastfm-360K/test.csv")
        genders = pd.read_csv("./datasets/Lastfm-360K/sensitive_attribute_random.csv")


    all_users = np.array(genders['user_id'].unique()).tolist()
    
    sensitive_attribute_0_num = genders[genders["gender"] == 0].shape[0]
    sensitive_attribute_1_num = genders[genders["gender"] == 1].shape[0]
    sensitive_attributes = np.array(genders["gender"].values)
    mark_0 = 0
    mark_1 = 0
    train_label_users = []
    for user,sensitive_attribute in zip(all_users, sensitive_attributes):
        if sensitive_attribute == 0 and mark_0 < int(sensitive_attribute_0_num *  male_know_size):
            mark_0 += 1
            train_label_users.append(user)
        if sensitive_attribute == 1 and mark_1 < int(sensitive_attribute_1_num *  female_know_size):
            mark_1 += 1
            train_label_users.append(user)
    train_nolabel_users = [user for user in all_users if user not in train_label_users]
    train_label_data = train_df.loc[train_df['user_id'].isin(train_label_users),]
    train_nolabel_data = train_df.loc[train_df['user_id'].isin(train_nolabel_users),]

    
  
    return train_df, train_label_data, train_nolabel_data, valid_df, test_df, train_label_users, train_nolabel_users


data_path = os.path.join(args.data_path, args.dataset) + "/"
train_data = pd.read_csv(data_path + "train.csv")
valid_data = pd.read_csv(data_path + "valid.csv")
test_data = pd.read_csv(data_path + "test.csv")
orig_sensitive_attr = pd.read_csv(data_path + "sensitive_attribute.csv")
genders = pd.read_csv(data_path + "sensitive_attribute_random.csv")


#80% for training in user

train_df, train_label_data, train_nolabel_data, valid_df, test_df, train_label_users, train_nolabel_users = process_data(args.dataset,
                                                                                                                         args.partial_ratio_male,
                                                                                                                         args.partial_ratio_female)
# generating sensitive attr mask for training

mark_0,mark_1 = 0,0
genders_for_train = genders[(genders["user_id"].isin(train_label_users))]
sensitive_attribute_0_num = genders_for_train[genders_for_train["gender"] == 0].shape[0]
sensitive_attribute_1_num = genders_for_train[genders_for_train["gender"] == 1].shape[0]
sensitive_attributes = np.array(genders_for_train["gender"].values)
train_users = []
for user,sensitive_attribute in zip(train_label_users, sensitive_attributes):
    if sensitive_attribute == 0 and mark_0 < int(sensitive_attribute_0_num * 0.8):
        mark_0 += 1
        train_users.append(user)
    if sensitive_attribute == 1 and mark_1 < int(sensitive_attribute_1_num * 0.8):
        mark_1 += 1
        train_users.append(user)
val_users = [user for user in train_label_users if user not in train_users]


orig_unfair_model_path = args.orig_unfair_model + "/" + args.dataset + "/" + "MF_orig_model"
orig_model = torch.load(orig_unfair_model_path, map_location = torch.device("cpu"))
user_embedding = orig_model['user_emb.weight']
user_embedding = user_embedding.detach().to(device)
classifier_model = sst_pred(user_embedding.shape[1], 32, 2).to(device)
# construct training set

# gender_known_male 
gender_known_male = genders[(genders["user_id"].isin(train_users)) & (genders["gender"] == 0)].user_id.values 
gender_known_female = genders[(genders["user_id"].isin(train_users)) & (genders["gender"] == 1)].user_id.values


gender_known_male_tensor = torch.LongTensor(gender_known_male)
gender_known_female_tensor = torch.LongTensor(gender_known_female)
train_tensor = torch.cat([user_embedding[gender_known_male_tensor], user_embedding[gender_known_female_tensor]])
train_label = torch.cat([torch.zeros(gender_known_male_tensor.shape[0]), torch.ones(gender_known_female_tensor.shape[0])]).to(device)

test_known_male = genders[(genders["user_id"].isin(val_users)) & (genders["gender"] == 0)].user_id.values 
test_known_female = genders[(genders["user_id"].isin(val_users)) & (genders["gender"] == 1)].user_id.values
test_known_male_tensor = torch.LongTensor(test_known_male)
test_known_female_tensor = torch.LongTensor(test_known_female)

test_tensor = torch.cat([user_embedding[test_known_male_tensor], user_embedding[test_known_female_tensor]])
test_label = torch.cat([torch.zeros(test_known_male_tensor.shape[0]), torch.ones(test_known_female_tensor.shape[0])]).to(device)



optimizer_for_classifier = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)
loss_for_classifier = torch.nn.CrossEntropyLoss()

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
train_dataset = CustomDataset(train_tensor, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)


##
for i in tqdm(range(args.gender_train_epoch)):
    for train_input, labels in train_dataloader:
        train_pred = classifier_model(train_input)
        loss_train = loss_for_classifier(train_pred, labels.type(torch.LongTensor).to(device))
        optimizer_for_classifier.zero_grad()
        loss_train.backward()
        optimizer_for_classifier.step()




test_acc, test_pred_male_female_ratio = evaluation_gender(test_tensor, test_label, classifier_model)





pred_all_label = classifier_model(user_embedding).max(1).indices


pred_all_label[gender_known_male] = 0
pred_all_label[gender_known_female] = 1



csv_folder = os.path.join(workspace, "sst_pred", args.dataset)
os.makedirs(csv_folder, exist_ok= True)
file_name  = f'{args.dataset}_maleratio_{args.partial_ratio_male}_femaleratio_{args.partial_ratio_female}_seed_{args.seed}_gender_train_epoch{args.gender_train_epoch}.csv'
save_path = os.path.join(workspace, "sst_pred",args.dataset, file_name)
pred_sensitive_attr = pd.DataFrame(list(zip(list(range(len(genders))), list(pred_all_label.cpu().tolist()))),\
     columns = ["user_id", "gender"])
pred_sensitive_attr.to_csv(save_path, index = None)

# save gammas
test_acc, gammas = evaluation_gender_new(test_tensor,test_label,classifier_model)
print(gammas)
print(test_acc)
gammas["acc"] = test_acc 
file_name = f'gammas_{args.dataset}_maleratio_{args.partial_ratio_male}_femaleratio_{args.partial_ratio_female}_seed_{args.seed}_gender_train_epoch{args.gender_train_epoch}.json'
save_path = os.path.join(workspace, "gammas", args.dataset, file_name)
os.makedirs(os.path.join(workspace, "gammas", args.dataset), exist_ok= True)
with open(save_path, "w+") as f:
    json.dump(gammas, f)
