import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random


### version 1.0
def get_cluster(bio_emb: torch.tensor, set_index: np.array):
    cluster_center = []
    set_start = set_index[0]
    with torch.no_grad():
        for set_end in set_index[1:-1]:
            current_cluster = bio_emb[set_start: set_end]
            cluster_center.append(current_cluster.mean(dim = 0, keepdim= True).detach().cpu())
            set_start = set_end
        
        # 终止边界处理
        # print(set_start)
        current_cluster = bio_emb[set_start: ]
        cluster_center.append(current_cluster.mean(dim = 0, keepdim= True).detach().cpu())
    return torch.cat(cluster_center).to(bio_emb.device) # [Gos, emb_dim]

def get_dist_map(bio_emb: torch.tensor, set_index: np.array, model = None):
    # updatate embeddings
    if model != None:
        bio_emb = model(bio_emb)

    center_emb = get_cluster(bio_emb, set_index)
    dist = torch.zeros((center_emb.shape[0], center_emb.shape[0])).to(bio_emb.device)
    for i, current in enumerate(center_emb):
        current = current.unsqueeze(0)
        dist[i] = (current-center_emb).norm(dim=1, p=2)
        
    return dist


### version 2.0
def get_cluster2(bio_emb: torch.tensor, set_index: np.array):
    cluster_centers = torch.zeros(len(set_index)-1, bio_emb.shape[-1]).to(bio_emb.device)
    set_start = set_index[0]
    with torch.no_grad():
        for i, set_end in enumerate(set_index[1:-1]):
            current_cluster = bio_emb[set_start: set_end]
            cluster_centers[i] = current_cluster.mean(dim = 0, keepdim= True)
            set_start = set_end
        
        # 终止边界处理
        # print(set_start)
        current_cluster = bio_emb[set_start: ]
        cluster_centers[-1] = current_cluster.mean(dim = 0, keepdim= True)
    return cluster_centers # [Gos, emb_dim]

def get_dist_map2(bio_emb: torch.tensor, set_index: np.array, model = None):
    # updatate embeddings
    if model != None:
        bio_emb = model(bio_emb)

    center_emb = get_cluster2(bio_emb, set_index)
    dist = torch.zeros((center_emb.shape[0], center_emb.shape[0])).to(bio_emb.device)
    
    for i, current in enumerate(center_emb):
        current = current.unsqueeze(0)
        dist[i] = (current-center_emb).norm(dim=1, p=2)
        
    return dist

def get_dist_map_test(model_emb_train: torch.tensor, model_emb_test: torch.tensor, set_index: np.array):
    center_emb = get_cluster2(model_emb_train, set_index)
    dist = torch.zeros((model_emb_test.shape[0], center_emb.shape[0])).to(model_emb_test.device)

    #print(f'center_mbeding:{center_emb.device}' )
    for i, current in enumerate(model_emb_test):
        current = current.unsqueeze(0)
        dist[i] = (current-center_emb).norm(dim=1, p=2)
        
    return dist

def mine_hard_negative(dist_map: torch.tensor, knn = 10):
    # negative cluster
    
    bottom_dist, index = dist_map.topk(knn+10, dim= -1, largest=False)
    bottom_dist, index = bottom_dist.detach().cpu().numpy(), index.detach().cpu().numpy()
    negtives = {}
    # 判断两类距离大于0
    judgement = bottom_dist!=0
    # 取前knn
    for i in range(bottom_dist.shape[0]):
        neg_go_index = index[i][judgement[i]][:knn]
        prob = np.asarray([1/dist for dist in bottom_dist[i][judgement[i]][:knn]])
        norm_prob = prob/prob.sum()
        negtives[i] = {'weights': norm_prob,'negative': neg_go_index}
    
    return negtives

def mine_negative(anchor_go_id: int, mine_neg: dict, set_index: np.array):
    # mine_neg: {go_id: {'weights': norm_prob,'negative': neg_go_index} }
    
    neg_go_ids = mine_neg[anchor_go_id]['negative']
    weights = mine_neg[anchor_go_id]['weights']
    neg_go_id = random.choices(neg_go_ids, weights=weights, k=1)[0]

    # sampling from current go term
    set_start, set_end = set_index[neg_go_id], set_index[neg_go_id+1]
    # print(neg_go_id, set_start, set_end)
    sample_id = np.random.randint(set_start, set_end)
    
    return sample_id

def random_positive(go_id: int, sample_id: int, set_index: np.array):

    set_start, set_end = set_index[go_id], set_index[go_id+1]
    # print(go_id, set_start, set_end)
    pos_sample_id = np.random.randint(set_start, set_end)
    
    while pos_sample_id == sample_id:
        pos_sample_id = np.random.randint(set_start, set_end)
    
    return pos_sample_id


class Triplet_dataset_with_mine_Go(torch.utils.data.Dataset):

    def __init__(self, mine_neg, bio_emb, sample2go_id, set_index):
        # self.gos = gos # np.array
        # self.go2id = go2id # dict: {go: go_id}
        self.mine_neg = mine_neg # dict: {go_id: {'weights': norm_prob,'negative': neg_go_index}}
        self.sample2go_id = sample2go_id # dict: {sample_id: go_id}
        self.set_index = set_index # np.array 
        self.bio_emb = bio_emb


    def __len__(self):
        return self.bio_emb.shape[0]

    def __getitem__(self, index):
        anchor_go_id = self.sample2go_id[index]
        pos_id = random_positive(anchor_go_id, index, self.set_index)
        neg_id = mine_negative(anchor_go_id, self.mine_neg, self.set_index)

        anchor_emb = self.bio_emb[index]
        pos_emb = self.bio_emb[pos_id]
        neg_emb = self.bio_emb[neg_id]
        
        return anchor_emb, pos_emb, neg_emb


class MultiPosNeg_dataset_with_mine_Go(torch.utils.data.Dataset):

    def __init__(self, mine_neg, bio_emb, sample2go_id, set_index, n_pos, n_neg):
        # self.gos = gos # np.array
        # self.go2id = go2id # dict: {go: go_id}
        self.mine_neg = mine_neg # dict: {go_id: {'weights': norm_prob,'negative': neg_go_index}}
        self.sample2go_id = sample2go_id # dict: {sample_id: go_id}
        self.set_index = set_index # np.array 
        self.bio_emb = bio_emb
        self.n_pos = n_pos
        self.n_neg = n_neg

    def __len__(self):
        return self.bio_emb.shape[0]

    def __getitem__(self, index):
        anchor_go_id = self.sample2go_id[index]
        combined_ids = [index]
        for  _  in range(self.n_pos):
            pos_id = random_positive(anchor_go_id, index, self.set_index)
            combined_ids.append(pos_id)
        for _ in range(self.n_neg):
            neg_id = mine_negative(anchor_go_id, self.mine_neg, self.set_index)
            combined_ids.append(neg_id)

        combined_emb = self.bio_emb[combined_ids] # [1+n_pos+n_neg, emb_dim]
        
        return combined_emb





# evaluate
## pvalue
def random_nk_model(emb_training, sample2go_id, n=10):
    nk = n*1000
    random_nk_id = np.random.choice(range(emb_training.shape[0]), nk, replace=False)
    random_nk_id = np.sort(random_nk_id)
    chosen_gos = [sample2go_id[go_id] for go_id in random_nk_id]

    chosen_emb_train = emb_training[random_nk_id]
    return chosen_gos, chosen_emb_train

def get_random_nk_dist_map(model_emb_train: torch.tensor, rand_nk_emb_train: torch.tensor, set_index: np.array, device: torch.device, go_weight: np.array, n=10):
    center_emb = get_cluster2(model_emb_train, set_index)
    dist = torch.zeros((rand_nk_emb_train.shape[0], center_emb.shape[0])).to(device)
    for i, current in enumerate(rand_nk_emb_train):
        current = current.unsqueeze(0)
        dist[i] = (current-center_emb).norm(dim=1, p=2)
        
    return dist

def pvalue_choices(dist_map: torch.tensor, true_label: torch.tensor,random_nk_dist_map: torch.tensor, pvalue=1e-5):
    # dist_map: [num_of_test, num_of_cluster]
    # random_nk_dist_map: [nk, num_of_cluster]
    nk = random_nk_dist_map.shape[1]
    threshold = pvalue*nk
    dists, index = dist_map.topk(10, dim= -1, largest=False)
    random_nk_dist_map_for_comparing = random_nk_dist_map.transpose(0, 1).sort(dim=-1)

    result = {}
    for i, label in enumerate(true_label):
        # top1 or distance <= threshold GO will be selected
        result[label] = [index[i][0].item()]
        tmp_random_dis_map = random_nk_dist_map_for_comparing[index[i][1:]]
        tmp_dist = (dists[i][1:]).unsqueeze(-1)
        rank = torch.searchsorted(tmp_random_dis_map, tmp_dist).reshape(-1) # [tok -1]
        filtered = rank <= threshold # bool [tok -1]
        result[label] += index[i][1:][filtered]

    return result

## topk
def topk_cluster(dist_map: torch.tensor, true_label: torch.tensor, top_k = 100):
    # only for single label
    # true_label: [num_of_test]
    # dist_map: [num_of_test, num_of_clusters]
    _, index = dist_map.topk(top_k, dim= -1, largest=False)
    precision = {}
    for i in [1, 5, 10, 20, top_k * 0.5, top_k]:
        bool_tensor = true_label.unsqueeze(-1)==index[:,:i] # [num_of_test, i]
        num_of_true = bool_tensor.count_nonzero().item() # num_of_test
        total_num = bool_tensor.numel()
        precision[i] = num_of_true/total_num
    return precision


#####
if __name__ == "__main__":
    paths = ['/media/linux/backup/qinbowen/biobert-relation-extraction-main/embeding_test/data_processed/dataset_v3.0_embedding/biobert/inner_trainingset_mean.pt', '/media/linux/backup/qinbowen/biobert-relation-extraction-main/embeding_test/data_processed/dataset_v3.0_embedding/biogpt/inner_trainingset_mean.pt']
    for path in paths:
        bio_emb = torch.load(path).cuda()
        # gos = np.loadtxt('/media/linux/backup/qinbowen/biobert-relation-extraction-main/embeding_test/data_prepare/go_data/go_obo_process.tsv',dtype=str, usecols=0)
        gos = np.loadtxt('/media/linux/backup/qinbowen/biobert-relation-extraction-main/embeding_test/data_prepare/go_data_v3.0/gos', dtype=str)
        set_index = np.loadtxt('/media/linux/backup/qinbowen/biobert-relation-extraction-main/embeding_test/data_prepare/go_data_v3.0/set_index',dtype=int)

        # clu_center = get_cluster(bio_emb, set_index)
        dis_map = get_dist_map(bio_emb, set_index).detach().cpu()
        print(dis_map.shape)
        # torch.save(clu_center, path.replace('total_mean.pt','clu_center.pt'))
        torch.save(dis_map, path.replace('mean.pt','dis_map.pt'))



