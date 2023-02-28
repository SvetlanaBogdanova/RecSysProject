import pandas as pd
import numpy as np
from math import log2


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)
    
    flags = np.isin(recommended_list[:k], bought_list)
    
    recall = (flags*prices_bought).sum() / prices_bought.sum()
    
    return recall


def mrr_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list)
    
    for i in range(k):        
        if flags[i]:
            return 1 / (i + 1)

    return 0


def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list)
    
    dcg = 0
    idcg = 0
    for i in range(k):        
        if flags[i]:
            dcg += 1 / log2(i + 2)
        if i < len(bought_list):
            idcg += 1 / log2(i + 2)
            
    return dcg / idcg