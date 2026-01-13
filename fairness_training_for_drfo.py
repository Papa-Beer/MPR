
import argparse
import numpy as np
import pandas as pd
import json 
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, ndcg_score, recall_score
from evaluation import evaluate_model_performance_and_naive_fairness_fast_rmse, evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse
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
import torch.nn.functional as F
import time
from collaborative_models import matrixFactorization, sst_pred
import warnings 
warnings.filterwarnings("ignore")
from tqdm import tqdm
def process_data(data_name, male_know_size, female_know_size,verbose=False):

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
    if verbose:
        print("train_label_users", len(train_label_users))
        print("train_nolabel_user", len(train_nolabel_users))
    
  
    return train_df, train_label_data, train_nolabel_data, valid_df, test_df, train_label_users, train_nolabel_users
def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

def project_simplex(v, z=1.0, axis=-1,):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = (torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.).long()
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


def project_onto_l1_ball_my(v, z = 1.0, axis = -1):
  if torch.norm(v, p = 1) <= z:
    w = v
  else:
    v_sign = torch.sign(v)
    v_abs = torch.abs(v)
    w_abs = project_simplex(v_abs, z, axis)
    w = torch.mul(w_abs, v_sign)
  return w
# DRFO Implementation 

class model_dro(nn.Module):
    def __init__(self, data_name,  male_ratio, female_ratio, init_lambdas=10,seed=1,gender_train_epoch=1000,orig_unfair_model=None,device=None, dataset_path=None, workspace_path = None):
        super().__init__()

        train_data, train_label_data, train_nolabel_data, valid_data, test_data, train_label_users,train_nolabel_users = process_data(data_name, male_ratio,female_ratio)
        self.train_label_users = train_label_users 
        self.train_nolabel_users = train_nolabel_users 
        self.train_label_data = train_label_data 
        self.train_nolabel_data = train_nolabel_data 
        self.device = device 
        df_train = train_data 
        self.train_data = train_data 
        self.valid_data = valid_data 
        self.test_data = test_data 
        self.batch_size = len(self.train_data)
        # sst prediction label position :)
        genders = pd.read_csv(os.path.join(dataset_path, data_name, "sensitive_attribute_random.csv"))
        pred_sst_df_path = f"{workspace_path}/sst_pred/{data_name}/{data_name}_maleratio_{male_ratio}_femaleratio_{female_ratio}_seed_{seed}_gender_train_epoch{gender_train_epoch}.csv"
        pred_sst_df = pd.read_csv(pred_sst_df_path)
        gamma_path = f"{workspace_path}/gammas/{data_name}/gammas_{data_name}_maleratio_{male_ratio}_femaleratio_{female_ratio}_seed_{seed}_gender_train_epoch{gender_train_epoch}.json"
        with open(gamma_path,"r+") as file:
          gamma_file = json.load(file)
          gammas = [gamma_file["0"],gamma_file["1"]]
          self.gammas = gammas 
        gender_map = {id:gender for (id,gender) in zip(genders["user_id"].values,genders["gender"].values)}
        train_label_male,train_label_female = [],[]
        for user in train_label_users:
            if gender_map[user] == 0:
                train_label_male.append(user)
            else:
                train_label_female.append(user)
        self.train_label_male = train_label_male
        self.train_label_female = train_label_female
        train_known_male_data =  df_train[(df_train["user_id"].isin(train_label_male)) ]
        train_known_female_data = df_train[(df_train["user_id"].isin(train_label_female))]
        train_unknown = df_train[df_train["user_id"].isin(train_nolabel_users)]
        pred_map = {id:gender for id,gender in zip(pred_sst_df["user_id"],pred_sst_df["gender"])}
        
        train_data["gender"] = train_data["user_id"].map(pred_map)
        train_unknown["gender"] = train_unknown["user_id"].map(pred_map)

        know_0 = len(train_known_male_data)
        know_1 = len(train_known_female_data)
        unknow_0 = len(train_unknown[train_unknown["gender"] == 0])
        unknow_1 = len(train_unknown[train_unknown["gender"] == 1])
        print("know_0",know_0)
        print("know_1",know_1)
        print("unknow_0",unknow_0)
        print("unknow_1",unknow_1)
        self.etas = {
            'know_0': know_0 / (know_0 + unknow_0),
            'know_1': know_1 / (know_1 + unknow_1), 
            'unknow_0': unknow_0 / (know_0 + unknow_0),
            'unknow_1': unknow_1/ (know_1 + unknow_1),
        }
        self.generate_user_group_distribution(self.train_data,max(self.train_data["user_id"].values) + 1,train_nolabel_users)
        print("etas",self.etas)
        
        sensitive_num = 1

        self.user_embedding_dict_dro = nn.ModuleDict({})
        self.user_embedding_dict_observed_noisy = nn.ModuleDict({})
        self.lambda_dict = nn.ParameterDict({str(i):nn.Parameter(torch.ones(1).to(device) * init_lambdas) for i in range(sensitive_num + 1)})
        # observed_group_distribution should be sorted in order for user 0, 1, 2, 3, 4 ...
        for i in range(sensitive_num + 1):
            # get dro_probability and save as user_embedding
            embedding_dict = nn.ModuleDict({})
            user_embedding = torch.from_numpy(self.group_distribution_dict[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict['user_id'] = nn.Embedding.from_pretrained(user_embedding, freeze = False).to(device)
            self.user_embedding_dict_dro[str(i)] = embedding_dict
            # get obser_probability

            embedding_dict2 = nn.ModuleDict({})
            user_embedding2 = torch.from_numpy(self.group_distribution_dict[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict2['user_id'] = nn.Embedding.from_pretrained(user_embedding2, freeze = True).to(device)
            self.user_embedding_dict_observed_noisy[str(i)] = embedding_dict2 
        
        self.user_emb = nn.Embedding.from_pretrained(orig_unfair_model["user_emb.weight"],freeze=False).to(device)
        self.item_emb = nn.Embedding.from_pretrained(orig_unfair_model["item_emb.weight"],freeze=False).to(device)
        self.user_bias = nn.Embedding.from_pretrained(orig_unfair_model["user_bias.weight"],freeze=False).to(device)
        self.item_bias = nn.Embedding.from_pretrained(orig_unfair_model["item_bias.weight"],freeze=False).to(device)
        self.out_act = nn.Sigmoid()
        
        
    


    def generate_user_group_distribution(self,train_data_with_label, total_user_num, generate_users):
        if len(generate_users):

            # set of unknown users
            generate_user_train_data = train_data_with_label[train_data_with_label['user_id'].isin(generate_users)]
            # data related to the generate users 
            sensitive_groups_dict = dict(list(generate_user_train_data.groupby('gender')))
            self.group_distribution_dict = {}
            keys = set()
            for key, group_df in sensitive_groups_dict.items():
                keys.add(key)
                if group_df.shape[0] != 0:
                    users_in_group = np.array(group_df['user_id'].unique()).tolist()
                    prob = 1 /  group_df.shape[0] #
                    emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                    self.group_distribution_dict[str(int(key))] = np.array(emp_probs)
                else:
                    users_in_group = np.array(group_df['user_id'].unique()).tolist()
                    prob = 0 # len(users_in_group)
                    emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                    self.group_distribution_dict[str(int(key))] = np.array(emp_probs)
            for g in [0,1]:
                if g not in keys:
                     emp_probs = [0 for user in range(total_user_num)]
                     self.group_distribution_dict[str(g)] = np.array(emp_probs)
        else:
            self.group_distribution_dict = {}
            emp_probs = [0 for user in range(total_user_num)]
            self.group_distribution_dict[str(0)] = np.array(emp_probs)
            self.group_distribution_dict[str(1)] = np.array(emp_probs)
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        out = self.out_act((U*V).sum(1) +  b_u  + b_v)
        return out
    def  get_dro_distribution_probability(self,u,i):
        return self.user_embedding_dict_dro[str(i)].user_id(u)
    def  get_dro_distribution_probability_noisy(self,u,i):
        return self.user_embedding_dict_observed_noisy[str(i)].user_id(u)       




def pretrain_epochs_with_predicted_sst_reg_eval_unfairness_valid_partial_rmse_thresh_drfo(model, 
                                                                         optim1,
                                                                         optim2,
                                                                         df_train, 
                                                                         epochs, 
                                                                         lr, 
                                                                         weight_decay, 
                                                                         batch_size, 
                                                                         valid_data, 
                                                                         test_data, 
                                                                         predicted_sensitive_attr, 
                                                                         oracle_sensitive_attr, 
                                                                         top_K, 
                                                                         fair_reg, 
                                                                         gender_known_male, 
                                                                         gender_known_female, 
                                                                         device, 
                        
                                                                         evaluation_epoch=10, 
                                                                         unsqueeze=False, 
                                                                         shuffle=True,
                                                                         early_stop=None,
                                                                         rmse_thresh = None):
    print(model.named_parameters())
    thetas = []
    batch_size = len(model.train_data)
    for name, param in model.named_parameters():
        if 'user_embedding' not in name and 'lambda' not in name:
            thetas.append(param)
            #print(name)
            print("theta",name)
    # thetas include user_emb,item_emb,user_bias,item_Bias      
    criterion = nn.BCELoss() 
    train_label_users = torch.LongTensor(model.train_label_users).squeeze().to(device)
    train_nolabel_users =  torch.LongTensor(model.train_nolabel_users).squeeze().to(device)
    male_known_users = torch.LongTensor(model.train_label_male).squeeze().to(device)
    female_known_users =  torch.LongTensor(model.train_label_female).squeeze().to(device)
    best_val_ndcg = 0 
    best_val_rmse = 100
    test_ndcg_in_that_epoch = 0
    val_UAUC_in_that_epoch = 0
    test_UAUC_in_that_epoch = 0
    best_epoch = 0
    naive_unfairness_val_in_that_epoch = 0
    naive_unfairness_test_in_that_epoch = 0
    best_val_unfairness = 1000
    for epoch in tqdm(range(epochs)):
        j = 0
        loss_total = 0
        random_id = np.array([_ for _ in range(len(df_train))])
        if shuffle:
            np.random.shuffle(random_id)                       
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size): # DATA batch
            # data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            data_batch = df_train.loc[random_id[batch_i:(batch_i+batch_size)]].reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            # train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            train_ratings = torch.FloatTensor(np.array(data_batch["label"])).to(device)
            train_user_input = torch.LongTensor(np.array(data_batch["user_id"])).to(device)
            train_item_input = torch.LongTensor(np.array(data_batch["item_id"])).to(device)
            train_user_sst =  torch.Tensor(np.array(predicted_sensitive_attr.iloc[np.array(data_batch["user_id"])]["gender"])).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings.view(-1))
            optim1.zero_grad()
            optim2.zero_grad()
            loss_total += loss.item()
            fair_constraints = {}
            sensitive_range = [1]
            label_user_index = torch.isin(train_user_input.squeeze(), torch.tensor(train_label_users, device = device))
            nolabel_user_index = torch.isin(train_user_input.squeeze(), torch.tensor(train_nolabel_users, device = device))
            sum_fair_constraint = 0
            for i in sensitive_range: 
              group_i_prob = model.get_dro_distribution_probability(train_user_input, i).squeeze() 


              if i == 0:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_hat).squeeze() -\
                model.etas['unknow_0'] * torch.sum(torch.mul(y_hat.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                model.etas['know_0'] * torch.mean(y_hat.squeeze()[(train_user_sst == i) & label_user_index])) 

              if i == 1:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_hat).squeeze() -\
                model.etas['unknow_1'] * torch.sum(torch.mul(y_hat.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                model.etas['know_1'] * torch.mean(y_hat.squeeze()[(train_user_sst == i) & label_user_index])) 

            
              sum_fair_constraint += model.lambda_dict[str(i)] * fair_constraints[str(i)]

            (-sum_fair_constraint).backward(inputs = list(model.user_embedding_dict_dro.parameters()), retain_graph = True)
            (loss + sum_fair_constraint).backward(inputs = thetas)
            optim1.step()
            optim2.step()
            with torch.no_grad():
               # set to 0
               for i in sensitive_range: 
                mask = torch.zeros_like(model.user_embedding_dict_dro[str(i)]['user_id'].weight.data).to(device)
                mask.scatter_(dim=0, index=train_nolabel_users.unsqueeze(1), value=1)
                model.user_embedding_dict_dro[str(i)]['user_id'].weight.data *= mask 

               for i in sensitive_range: 
                  u_sort,_ = torch.sort(train_user_input.squeeze())
                  no_label_index = torch.isin(u_sort, torch.tensor(train_nolabel_users, device = device))
                  u_sort = u_sort[no_label_index].reshape(-1,1)
                  p_dro = model.get_dro_distribution_probability(u_sort, i).squeeze()
                  p_noisy = model.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                  u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                  cnt = pd.value_counts(u_sort_df)
                  indices = []
                  index = 0
                  for idx in cnt.index.unique().sort_values().tolist():
                    indices.append(index)
                    index += cnt[idx]
                  unique_index = torch.unique(u_sort.squeeze()).long()
 
                  p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * model.gammas[i])  + p_noisy).squeeze()

                  iter_0, iter_1 = 0,0
                  eps = 1e-5
                  while True:
                    p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                    p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * model.gammas[i])  + p_noisy).squeeze()
                    if i == 0:
                        iter_0 += 1
                    if i == 1:
                        iter_1 += 1
                    if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                        p_proj = p_proj_l1.reshape(-1, 1)
                        break
                    if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                        p_proj = p_proj_l1.reshape(-1, 1)
                        eps /= 0.9
                        break
                  model.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]

            j = j+1
        print('epoch: ', epoch, 'average loss: ',loss_total/ j)

        if epoch % evaluation_epoch ==0 :
            t0 = time.time()
            rmse_val, naive_unfairness_val = evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse(model, valid_data, oracle_sensitive_attr, male_known_users, female_known_users, top_K, device)
            t1 = time.time()
            rmse_test, naive_unfairness_test = evaluate_model_performance_and_naive_fairness_fast_rmse(model, test_data, oracle_sensitive_attr, top_K, device)
            t2 = time.time()
            print('epoch: ', epoch, 'val RMSE', rmse_val, 'Partial Valid Unfairness:', naive_unfairness_val, " time:" , str(t1 - t0))
            print('epoch: ', epoch, 'test RMSE', rmse_test, "Unfairness:", naive_unfairness_test, " time:", str(t2 - t1))

            if rmse_val < rmse_thresh:
                if naive_unfairness_val < best_val_unfairness:
                    val_rmse_in_that_epoch = rmse_val
                    test_rmse_in_that_epoch = rmse_test
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    best_val_unfairness = naive_unfairness_val
                    naive_unfairness_test_in_that_epoch = naive_unfairness_test

    return val_rmse_in_that_epoch, test_rmse_in_that_epoch, best_val_unfairness, naive_unfairness_test_in_that_epoch, best_epoch, best_model

