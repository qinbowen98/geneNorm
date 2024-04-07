import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoModel
import multiprocessing as mp
import os
import argparse
from pipeline_intergrate import intergrate_pipeline
from pipeline_api import multi_search_and_save, geneid_concat
import os.path as osp
import json5
from model import LayerNormNet
from dataloader import get_dist_map_test, get_cluster2
import pandas as pd
import sys
from transformers import logging
import urllib3
import argparse
import time
import ssl
import warnings
import re
import json
from Bio import Medline, Entrez


base_path = "/data/services/biology/python/geneNorm"#改成放该文件夹的地址
GeneTaggerV2Path = "/data/services/biology/python/geneTaggerV2"

def multi_search_and_save_genetagger(df,save_file_string):
    #df = pd.read_csv(gene_species_file,sep='\t')
    df = df
    pmids = []
    genes = []
    species = []
    geneids = []
    total = 0

    for i in range( len(df)):
        total += 1
        inds = {}
        if df['gene'][i] != 'None' and df['species'][i] != 'None':
            sp = str(df['species'][i])
            gene = str(df['gene'][i])
            for s in sp.split(" "):
                gene = gene.replace(s,"")
            pmids.append(str(df['pmid'][i]))
            genes.append(str(df['gene'][i]))
            species.append(str(df['species'][i]))
            temp = []
            '''
            for trys in range(1):
                records = esearch_species(str(df["species"][i]))
                print("thread ", str(num), " trys_species ", trys, " ", records['IdList'])
                for j in records['IdList']:
                    r.append(str(j))
            if len(r) > 0:
                taxids.append("##".join(r))
            else:
                taxids.append("None")
            '''
            #all search
            flag = 1
            while flag:
                try:
                    for sps in sp.split("##"):
                        record_new = Entrez.read(
                                        Entrez.esearch(db="gene",
                                                       term=f"({str(gene)}[All Fields]" \
                                                            f" AND {str(sps)}) AND alive[prop]",
                                                       usehistory='y',sort = 'relevance'))
                        for s in record_new['IdList']:
                            #print(record_new['IdList'].index(s))
                            temp.append(s)
                            if s in inds.keys():
                                inds[s] = inds[s] + '|' + gene + "-" + sps + '-' + str(record_new['IdList'].index(s))
                            else:
                                inds[s] = gene + "-" + sps + '-' + str(record_new['IdList'].index(s))
                    flag = 0
                except:
                    flag = 1
            temp = list(set(temp))
            temp_all = []
            for s in temp:
                temp_all.append(s+'('+inds[s]+')')
            if len(temp) > 0:
                geneids.append(",".join(temp))
            else:
                geneids.append("None")
    df['gene_ids'] = geneids
    df.to_csv(save_file_string,sep='\t',index=False)

warnings.filterwarnings('ignore')

logging.set_verbosity_error()
urllib3.disable_warnings()

ssl._create_default_https_context = ssl._create_unverified_context
Entrez.email = '1847156239@qq.com'
Entrez.api_key = '73664c0b675e8450f3cf90add93e69820808'
outputfile = ""
outputdic = {}

# current_directory ="E:\GBT\pipeline\pipeline_genetaggerV2"# to haoran 这里改成后台存结果的地方，如果没有改文件存储路径的话就找pipeline_genetaggerV2所在目录
#current_directory = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
model_gene = GeneTaggerV2Path+"/biobert_model_ner_gene"
model_species = GeneTaggerV2Path+"/biobert_model_ner_specis"
model_token = GeneTaggerV2Path+"/biobert_v1.1_pubmed"


def entity_link_update_genetagger(function_file, save_file):
    pmid = []
    functions_all = []
    IDs = []
    GO = obo_parser.GODag("E:\GBT\pipeline\go.obo")
    tokenizer = BertTokenizer.from_pretrained(model_token, do_lower_case=False)
    tag_values = ['B', 'E', 'I', 'O', 'S']
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    model = BertForTokenClassification.from_pretrained(
        model_token,
        num_labels=len(tag2idx),
        output_attentions=False,
        output_hidden_states=True
    )
    start_time = time.time()
    df_function = pd.read_csv("go_mb_ours.csv")
    go_name_all = df_function["Name"].tolist()
    go_id_all = df_function["ID"].tolist()
    functions_length = []

    df_useful = list(set(go_name_all))
    function_features = np.loadtxt('entity_link_embedding_go_mb_dataset.csv', delimiter=',')
    #print("old ",len(go_name_all))
    #print("new ",len(df_useful))
    # In[ ]:
    function_features_clean = []
    for i in df_useful:
        function_features_clean.append(function_features[df_useful.index(i)])
    df_length = []
    for i in df_useful:
        df_length.append(len(tokenizer.encode(i)))
    min_function = min(df_length)
    max_function = max(df_length)
    #print(min_function,max_function)
    end_time = time.time()
    #print("load data ",end_time-start_time)

    model.eval()
    df = pd.read_csv(function_file, sep='\t')
    function_entity = df['function'].tolist()
    temp_length = int(len(df) / 8) + 1

    test_entities = ["expressed at high levels during active secondary wall cellulose synthesis in developing cotton fibers"#"GO:0030244"
                    ,"exhibits a transcriptionally regulated circadian rhythm"#"GO:0007623"
                     ,"the germination response of spores"#"GO:0009847"
                    ,"induced by jasmonic acid"#GO:0009753
                     ]
    for num in range(len(df)):
        '''
        if num % (int(len(df) / 10) + 1) == 0:
            print("entity_link_update:", num / len(df))
        '''
        start_time = time.time()
        function_entities = str(function_entity[num])
        function_entities = re.sub(r"\s*[^A-Za-z]+\s*", " ", function_entities)
        function_sims = []
        functions = []
        functions_name = []
        tokenized_sentence = tokenizer.encode(function_entities)
        # print(tokenized_sentence)
        # print(len(tokenized_sentence))
        # input_ids = torch.tensor([tokenized_sentence]).cuda()
        input_ids = torch.tensor([tokenized_sentence[:511]])
        # pmids_d.append(df1['pmid'][num])
        # all_stre.append(str(i))
        with torch.no_grad():
            output = model(input_ids)
        end_time = time.time()
        #print("produce embedding ", end_time - start_time)
        ''''''
        start_time = time.time()
        for i in range(len(df_function)):
            tokenized_function = tokenizer.encode(df_function["Name"][i])
            temp_length = min(len(tokenized_function) - 2, len(function_entities.split(" ")))
            if df_function['Depth'][i] >= 4 and GO.query_term(df_function['ID'][i]):
                go_embedding = function_features[i]
                sim = []
                temp = []
                # print(len(output.hidden_states[0][0][1:-1]))
                for l in range(len(output.hidden_states[0][0][1:-1]) - temp_length):
                    s = sum(output.hidden_states[0][0][l + 1:l + temp_length + 1]).detach().numpy().tolist()

                    # print(len(sum(output.hidden_states[0][0][1:-1]).detach().numpy().tolist()))
                    # print(len(output.hidden_states[0][0][-1].detach().numpy().tolist()))
                    sim.append(cos_sim(go_embedding, s))
                if len(sim) > 0:
                    if max(sim) > 0.70:
                        functions.append(df_function["ID"][i])
                        function_sims.append(max(sim))
                        functions_name.append(df_function["Name"][i])

        end_time = time.time()
        
        #print("slide ", end_time - start_time)

        start_time = time.time()
        hidden_states = output.hidden_states[0][0][1:-1].detach().numpy()
        function_entities_len = len(tokenizer.encode(function_entities))

        # 假设go_embedding已经是合适的形状，或者进行了适当的调整
        # 示例中未提供function_features的结构，假设它是一个二维数组
        # 生成一个包含所有滑动窗口的矩阵
        '''
        windows_all = []
        for i in range(min_function):
            windows_all.append([])
        for length in range(min_function,max_function+1):
            windows = []
            temp_l = max(1,min(length, function_entities_len)-4)
            print("temp length",temp_l)
            print("temp length", len(hidden_states) - temp_l)
            for l in range(len(hidden_states) - temp_l):
                #tokenized_function = tokenizer.encode(df_function["Name"][i])
                s = sum(output.hidden_states[0][0][l + 1:l + temp_l + 1]).detach().numpy().tolist()
                windows.append(s)
            windows = np.array(windows)
            windows_all.append(windows)
        '''
        windows_all = [np.array([hidden_states[l:l + temp_l] for l in range(len(hidden_states) - temp_l)])
                       for temp_l in range(1, max_function - min_function + 2)]
        #windows = np.array([hidden_states[l:l + function_entities_len] for l in range(len(hidden_states) - function_entities_len + 1)])
        #print("windows ",windows.shape)
        for i, row in df_function.iterrows():
            if row['Depth'] >= 4 and GO.query_term(row['ID']):
                temp = []
                tokenized_function = tokenizer.encode(df_function["Name"][i])
                for s in range(len(windows_all[len(tokenized_function)])):
                    temp.append(function_features[i])
                go_embeddings = np.array(temp)
                #print("windows ", go_embeddings.shape)
                # 对于每个满足条件的行，计算go_embedding与所有窗口的相似度
                #print(len(tokenized_function))
                #print(go_embeddings.shape, windows_all[len(tokenized_function)].shape)
                sim = vectorized_cos_sim(go_embeddings, windows_all[len(tokenized_function)])  # go_embeddings[i:i+1] 保持二维形状
                max_sim = np.max(sim)

                if max_sim > 0.70:
                    functions.append(row["ID"])
                    function_sims.append(max_sim)
                    functions_name.append(row["Name"])

        end_time = time.time()
        #print("slide new", end_time - start_time)
        start_time = time.time()




        result = []
        for i in functions_name:
            id_temp = functions[functions_name.index(i)]
            parents = GO.query_term(id_temp).get_all_parents()
            p = []
            for G in parents:
                p.append((G, GO.query_term(G).depth))
            p.sort(key=takeSecond)
            flag = 0
            for j in p:
                temp_max = 0
                temp_lab = 0
                if j[0] in functions:
                    for k in range(len(functions)):
                        if function_sims[k] > temp_max and functions[k] == j:
                            temp_max = function_sims[k]
                            temp_lab = k
                    result.append((functions_name[k], temp_max))
                    flag = 1
                    break
            if flag == 0:
                result.append((i, function_sims[functions_name.index(i)]))
        result = list(set(result))
        result.sort(key=takeSecond, reverse=True)

        end_time = time.time()
        #print("GO assemble ", end_time - start_time)
        start_time = time.time()
        if len(result) > 0:
            node_queue = [result[0]]
            for pairs in result:
                flag = 0
                for j in node_queue:
                    if cos_sim(function_features[go_name_all.index(j[0])],
                               function_features[go_name_all.index(pairs[0])]) > 0.7:
                        flag = 1
                if flag == 0:
                    node_queue.append(pairs)
                    node_queue = list(set(node_queue))
            node_queue.sort(key=takeSecond, reverse=True)
            temp = []
            for s in node_queue[:3]:
                temp.append(go_id_all[go_name_all.index(s[0])])
            IDs.append(",".join(temp))
        else:
            IDs.append("None")
        #       print("find: ", s)
        # print("#################")
        #print("final assemble ", end_time - start_time)
    df["old_link"] = IDs
    df.to_csv(save_file, sep='\t')




def tsv_reader(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

class EntityLinker():
    """
    实体链接器类，用于链接生物学实体。
    """
    def __init__(self, database_path: str, saving_path: str, linker_option: str = 'pruning_60_5_v3', 
                 pretrained_model_path: dict = None, pretrained_model_type='biobert',
                 device=torch.device('cpu'), dtype=torch.float32):
        """
        初始化实体链接器。
        """
        self._validate_linker_option(linker_option)
        self.device, self.dtype = device, dtype
        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model, self.tokenizer = self._load_pretrained_model(pretrained_model_path, pretrained_model_type)
        self.cur_database_path = osp.join(database_path, linker_option)
        self.train_bio_emb_after, self.model = self._load_training_data_and_model(saving_path, linker_option, pretrained_model_type)
        self.set_index, self.sample2go_id, self.id2go = self._load_database_files()

    def _validate_linker_option(self, linker_option):
        """
        验证链接器选项。
        """
        valid_options = ['no_pruning', 'pruning_95_5_v3', 'pruning_90_5_v3', 'pruning_80_5_v3', 'pruning_60_5_v3', 'pruning_40_5_v3']
        if linker_option not in valid_options:
            raise ValueError(f'linker_option must be one of {valid_options}')

    def _load_pretrained_model(self, pretrained_model_path, pretrained_model):
        """
        载入预训练模型和分词器。
        """
        if pretrained_model == 'biobert':
            model_path = pretrained_model_path['biobert']
            model = AutoModel.from_pretrained(model_path).to(device=self.device, dtype=self.dtype)
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            raise ValueError('pretrained_model should be biobert or biogpt')
        return model, tokenizer

    def _load_training_data_and_model(self, saving_path, linker_option, pretrained_model):
        """
        载入训练数据和模型。
        """
        train_bio_emb_path = osp.join(self.cur_database_path, f'embed/{pretrained_model}/train.pt')
        train_bio_emb = {'training': torch.load(train_bio_emb_path).to(device=self.device, dtype=self.dtype)}
        model_checkpoint_path = osp.join(saving_path, linker_option, f'model/{pretrained_model}', 'cur_best_model.pth')
        model = LayerNormNet(train_bio_emb['training'].shape[-1], 512, 128, self.device, self.dtype)
        model.load_state_dict(torch.load(model_checkpoint_path,map_location=torch.device('cpu')))
        model.eval()
        train_bio_emb_after = model(train_bio_emb['training'])
        return train_bio_emb_after, model

    def _load_database_files(self):
        """
        载入数据库文件。
        """
        set_index_path = osp.join(self.cur_database_path, 'raw_data', 'set_index')
        sample2go_id_path = osp.join(self.cur_database_path, 'raw_data', 'sample2go_id')
        gos_path = osp.join(self.cur_database_path, 'raw_data', 'gos')
        
        set_index = np.loadtxt(set_index_path, dtype=int)
        sample2go_ids =np.loadtxt(sample2go_id_path, dtype=int, usecols=0)
        sample2go_id = {sample_id: go_id for sample_id, go_id in enumerate(sample2go_ids)}
        gos = np.loadtxt(gos_path, dtype=str)
        id2go = {id: go for id, go in enumerate(gos)}
        return set_index, sample2go_id, id2go

    def infer(self, input_str_list: list, topk=10, batch_size=None):
        """
        推断输入字符串列表中的实体。
        """
        if isinstance(input_str_list, str):
            input_str_list = [input_str_list]
        if batch_size is None:
            batch_size = len(input_str_list)
        
        input_embedding_list = self._get_input_embedding(input_str_list, batch_size)
        true_labels = self._predict_labels(input_embedding_list, topk)
        return true_labels

    def _get_input_embedding(self, input_str_list, batch_size):
        """
        获取输入字符串列表的嵌入表示。
        """
        self.pretrained_model.zero_grad(set_to_none=True)
        get_embedding = self.pretrained_model.get_input_embeddings()

        embed_list = []
        for batch in self._batchify(input_str_list, batch_size):
            abstract_tokens = self.tokenizer(batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(self.device)
            abstract_embedding = get_embedding(abstract_tokens).mean(dim=-2)
            embed_list.append(abstract_embedding.detach().cpu())
        return embed_list

    def _predict_labels(self, input_embedding_list, topk):
        """
        根据输入嵌入预测标签。
        """
        true_labels = []
        for abstract_embedding in input_embedding_list:
            emb_test = self.model(abstract_embedding)
            eval_dist = get_dist_map_test(self.train_bio_emb_after, emb_test, self.set_index)
            _, index = eval_dist.topk(topk, dim=-1, largest=False)

            for idx in index.detach().cpu():
                true_labels.append([self.id2go[go_idx] for go_idx in idx.tolist()])
        return true_labels

    def _batchify(self, input_list, batch_size):
        """
        将输入列表分批处理。
        """
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    def _get_clusters(self):
        """
        获取训练前后的嵌入中心。
        返回:
            center_emb_before: 训练前的嵌入中心。
            center_emb_after: 训练后的嵌入中心。
        """
        # 训练前的中心嵌入
        center_emb_before = get_cluster2(self.train_bio_emb_after, self.set_index)
        # 训练后的中心嵌入
        center_emb_after = get_cluster2(self.train_bio_emb_after, self.set_index)

        return center_emb_before, center_emb_after

    def embedding_plot(self, valid_embedding=None, valid_labels=None):
        """
        绘制有效嵌入和中心嵌入的可视化图。
        参数:
            valid_embedding: 可选，有效嵌入的数据。
            valid_labels: 可选，有效嵌入对应的标签。
        """
        # 获取训练前后的中心嵌入
        center_emb_before, center_emb_after = self._get_clusters()

        # 如果提供了有效嵌入，处理并转换这些嵌入
        if valid_embedding is None:
            valid_embedding_before_path = osp.join(self.cur_database_path, f'embed/{self.pretrained_model_type}/valid.pt')
            valid_embedding_before = torch.load(valid_embedding_before_path).to(device=self.device, dtype=self.dtype)
            valid_embedding_after = self.model(valid_embedding_before)
            valid_embedding_after = valid_embedding_after.detach().cpu().numpy()
        
        # 如果提供了有效标签，加载这些标签
        if valid_labels is None:
            valid_labels_path = osp.join(self.cur_database_path, 'raw_data', 'valid.tsv')
            valid_labels = []
            for line in tsv_reader(valid_labels_path):
                valid_labels.append(line.split('\t')[0])

        
        return center_emb_before, center_emb_after, self.id2go.values(), valid_embedding_before, valid_embedding_after, valid_labels


def find_tsv_files(root_dir, pattern):
    """
    遍历 root_dir 及其所有子目录，查找包含特定模式的 .tsv 文件。

    :param root_dir: 要遍历的根目录路径
    :param pattern: 文件名中要搜索的模式字符串
    :return: 包含符合条件文件路径的列表
    """
    matching_files = []  # 用于存储找到的文件路径

    # os.walk() 递归遍历 root_dir 及其所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 遍历当前目录下的所有文件名
        for filename in filenames:
            # 检查文件名是否包含指定的模式并且是 .tsv 文件
            if pattern in filename and filename.endswith('.tsv'):
                # 构造完整的文件路径并添加到列表中
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files



def main():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-p', '--ProcessID', default="998")
    parser.add_argument('-pn', '--NormProcessID')
    args = parser.parse_args()
    try:
        ProcessID = args.ProcessID
        NormProcessID = args.NormProcessID
    except Exception as e:
        outputdic["Reason data"] = repr(e)
        outputdic["Result data"] = False
        outputdic["Process ID"] = NormProcessID
        res = json5.dumps(outputdic)
        print(res)
        sys.exit(1)
    try:
        date = ""
        subject = GeneTaggerV2Path+"/"+ProcessID


        # 使用当前目录作为遍历的起点
        current_dir = subject  # '.' 表示当前目录
        pattern = 'gene_species_withfunction'

        # 查找并打印所有匹配的 .tsv 文件路径
        matching_files = find_tsv_files(subject, pattern)
        for file_path in matching_files:
            #print(file_path)
            
            database_path = osp.join(base_path, 'raw_data')
            saving_path = osp.join(base_path, 'models/contrastive_learning')

            pretrained_model_path = {
                'biobert': osp.join(base_path, 'models/pretrained/biobert_v1.1_pubmed_pytorch_model'),
            }
            linker_option = 'pruning_60_5_v3'
            #这个方法对应1万以上功能
            '''
            entity_linker = EntityLinker(database_path=database_path, saving_path=saving_path,
                                         pretrained_model_path=pretrained_model_path)
            df = pd.read_csv(file_path,sep='\t')
            goterms = []
            for i in range(len(df)):
                example = df["function"][i]
                result = entity_linker.infer(example, topk=5)
                goterms_name = ",".join(result[0])
                goterms.append(goterms_name)
            # 显示合并后的 DataFrame 的前几行
            '''
            final_temp_file = file_path.replace("gene_species_withfunction", "final_file_goterm")
            entity_link_update_genetagger(file_path,final_temp_file)
            df_temp = pd.read_csv(final_temp_file,sep='\t')
            df["GOterms"] = df_temp["old_link"]
            final_file = file_path.replace("gene_species_withfunction","final_file")

            multi_search_and_save_genetagger(df,final_file)
        all_final_file = subject + "/geneid_" + date + "_final.tsv"
        pattern = 'final_file'
        # 查找并打印所有匹配的 .tsv 文件路径
        matching_files = find_tsv_files(current_dir, pattern)
        dfs = [pd.read_csv(file_path, sep='\t') for file_path in matching_files]
        # 使用 pd.concat 来合并所有的 DataFrame 到一个单独的 DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(all_final_file, sep='\t', index=False)
        outputdic["Result data"] = True
        outputdic["Data json"] =all_final_file
        outputdic["Process ID"] = NormProcessID
        res = json.dumps(outputdic)
        print(res)
    except:
        import traceback
        outputdic["Reason data"] = traceback.format_exc()
        outputdic["Result data"] = False
        outputdic["Process ID"] = NormProcessID
        res = json5.dumps(outputdic)
        print(res)
if __name__ == '__main__':
    main()
