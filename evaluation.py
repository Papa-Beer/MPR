import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
import pandas as pd


def rooted_mean_squared_error(y_true, y_pred):
    # y_pred = np.array(x['predict_ratings'])
    # y_true = np.array(x['ratings'])
    return np.sqrt(mean_squared_error(y_true, y_pred))



# prob = y_hat
# labels = test_rating

# evaluation gender classifier
def evaluation_gender(data, label, model):
    model.eval()
    pred = model(data)
    pred_out = pred.argmax(1)
    acc = round(sum(pred_out == label).item()/(pred_out.shape[0]) * 100, 2)
    pred_male_female_ratio = ((sum(pred_out == 0).item() + 1e-2)/(sum(pred_out == 1).item() + 1e-2))
    return acc, pred_male_female_ratio

def evaluate_model_performance_and_naive_fairness_fast_partial_valid_rmse(model, df_val, df_sensitive_attr, gender_known_male, gender_known_female, top_K, device):
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        fairness_count = 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        y_true_all = []
        y_pred_all = []
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index]
            y_true_all += list(test_rating)
            y_pred_all += y_hat.tolist()
            if len(np.unique(test_rating)) != 1:
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                uniq_count += 1
            if (name in gender_known_male) or (name in gender_known_female): 
                gender = int(df_sensitive_dict.iloc[name]["gender"])
                naive_fairness_dict[gender] += y_hat.tolist()
                fairness_count += 1

        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
        rmse_result = rooted_mean_squared_error(y_true_all, y_pred_all)
    return rmse_result, naive_gender_unfairness




def evaluate_model_performance_and_naive_fairness_fast_rmse(model, df_val, df_sensitive_attr, top_K, device):
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"])).to(device)
        test_item_total = torch.tensor(np.array(df_val["item_id"])).to(device)
        pred_total = model(test_user_total, test_item_total)
        pred_total = pred_total.cpu().detach()
        uniq_count= 0
        naive_fairness_dict = {i:[] for i in df_sensitive_attr["gender"].unique()}
        df_val_with_index = df_val.reset_index()
        group_val = df_val_with_index.groupby("user_id")
        df_sensitive_dict = df_sensitive_attr.set_index("user_id")
        y_true_all = []
        y_pred_all = []
        for name, group in group_val:
            test_rating = np.array(group["label"]).astype(int)
            y_hat = pred_total[group.index] 
            y_true_all += list(test_rating)
            y_pred_all += y_hat.tolist()
            if len(np.unique(test_rating)) != 1:
                y_hat_sort_id = y_hat.sort(descending=True).indices
                label_rank = test_rating[y_hat_sort_id]
                uniq_count += 1
            gender = int(df_sensitive_dict.iloc[name]["gender"])
            naive_fairness_dict[gender] += y_hat.tolist()

        naive_gender_unfairness = float(np.abs(np.mean(naive_fairness_dict[1]) - (np.mean(naive_fairness_dict[0]))))
        rmse_result = rooted_mean_squared_error(y_true_all, y_pred_all)
    return rmse_result, naive_gender_unfairness

def evaluation_gender_new(data, label, model):
    # for drfo caculate gamma
    model.eval()
    pred = model(data)
    pred_out = pred.argmax(1)
    gammas = {}
    acc = round(sum(pred_out == label).item()/(pred_out.shape[0]) * 100, 2)
    gammas[0] = 1 - accuracy_score(pred_out[label == 0].cpu().numpy(), label[label == 0].cpu().numpy())
    gammas[1] = 1 - accuracy_score(pred_out[label == 1].cpu().numpy(), label[label == 1].cpu().numpy())
        
    return acc, gammas 
