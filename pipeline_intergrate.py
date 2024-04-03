import os
import sys
import xlrd
import codecs
import collections
import json
import io
from  collections import Counter
import string
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib
import pandas as pd
import nltk
import networkx as nx
import pandas as pd
import re
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from matplotlib_venn import venn2,venn3, venn3_circles
import heapq
from graphviz import Digraph
from ete3 import NCBITaxa
from goatools import obo_parser
def takeSecond(elem):
    return elem[1]

def intergrate_pipeline(save_file,geneid_gene_specie_name_matching_file,geneid_gene_specie_withfunction_file,geneid_gene_specie_matching_file,geneid_gene_specie_file):
    '''
    these file see the main pipelne
    '''
    GO =obo_parser.GODag("go.obo")
    print(GO.query_term("GO:0001510").name)
    #gene_candidates = ['a103r','d1r','d12l','vp39','vp4','ns3','ns5','l protein','vp3','np868','cet1','ceg1','abd1','mce1','rnmt','cmtr1','hce']
    df_go = pd.read_csv(geneid_gene_specie_withfunction_file,sep='\t')
    geneid_gene = {}
    geneid_species = {}
    geneid_iffunction = {}
    geneid_function_entitys = {}
    geneid_pmid = {}
    geneid_functions = {}
    geneids = []
    df_function = pd.read_csv(geneid_gene_specie_withfunction_file,sep='\t')
    print(len(df_function))
    for i in range(len(df_function)):
        if i % len(df_function)/10 == 0:
            print("loading result withfunction ",i/len(df_function))
        if str(df_function['gene_id'][i]) != 'None':
            for geneid in str(df_function['gene_id'][i]).split("##"):
                geneid = geneid.split("(")[0]
                if str(geneid) in geneid_functions.keys():
                    geneid_functions[str(geneid)] =  str(geneid_functions[str(geneid)])+"##"+str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(geneid_function_entitys[str(geneid)])+"##"+str(df_function['function'][i])+':'+str(df_function['pmid'][i])
                else:
                    geneid_functions[str(geneid)] =str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(df_function['function'][i])+':'+str(df_function['pmid'][i])
    df_function = pd.read_csv(geneid_gene_specie_matching_file,sep='\t')
    for i in range(len(df_function)):
        if i % len(df_function) / 10 == 0:
            print("loading result matching", i / len(df_function))
        if str(df_function['gene_id'][i]) != 'None':
            for geneid in str(df_function['gene_id'][i]).split("##"):
                geneid = geneid.split("(")[0]
                if str(geneid) in geneid_functions.keys():
                    geneid_functions[str(geneid)] =  str(geneid_functions[str(geneid)])+"##"+str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(geneid_function_entitys[str(geneid)])+"##"+str(df_function['function'][i])+"@maching@ "+':'+str(df_function['pmid'][i])
                else:
                    geneid_functions[str(geneid)] = str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(df_function['function'][i])+"@maching@ "+':'+str(df_function['pmid'][i])


    #gene Name
    df_function = pd.read_csv(geneid_gene_specie_name_matching_file,sep='\t')
    for i in range(len(df_function)):
        if i % len(df_function) / 10 == 0:
            print("loading result name matching", i / len(df_function))
        if str(df_function['gene_id'][i]) != 'None':
            for geneid in str(df_function['gene_id'][i]).split("##"):
                geneid = geneid.split("(")[0]
                if str(geneid) in geneid_functions.keys():
                    geneid_functions[str(geneid)] =  str(geneid_functions[str(geneid)])+"##"+str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(geneid_function_entitys[str(geneid)])+"##"+str(df_function['function'][i])+"@name@ "+':'+str(df_function['pmid'][i])
                else:
                    geneid_functions[str(geneid)] = str(df_function['ID'][i])
                    geneid_function_entitys[str(geneid)] = str(df_function['function'][i])+"@name@ "+':'+str(df_function['pmid'][i])



    df = pd.read_csv(geneid_gene_specie_file,sep='\t')
    print(len(df))
    '''
    df = df.dropna()
    df.reset_index()
    '''
    df_function = pd.read_csv(geneid_gene_specie_name_matching_file,sep='\t')
    cop = re.compile("[^a-z^A-Z^0-9]")
    genename_goterm = {}
    for i in range(len(df_function)):
        if i % 200 == 0:
            print("1 ",i)
            #f = str(df_function['GOterms'][i])
        gene_temp = cop.sub("",str(df_function['gene'][i]))
        if gene_temp in genename_goterm.keys():
            if len(str(genename_goterm[gene_temp]).split("##"))<5:
                genename_goterm[gene_temp]  = genename_goterm[gene_temp]+'##'+str(df_function['ID'][i])
        else:
            genename_goterm[gene_temp] = str(df_function['ID'][i])
    geneid_genename_goterm = {}
    for i in range(len(df)):
        if i % len(df) / 10 == 0:
            print("making dict ", i / len(df))

        gene_temp = cop.sub("", str(df['gene'][i]))
        if gene_temp in genename_goterm.keys():
            s = genename_goterm[gene_temp]
            s_t = []
        else:
            s = 'None'

        if str(df['gene_id'][i]) != 'None':
            for j in str(df['gene_id'][i]).split("##"):
                j = j.replace(")", "").split("(")
                if len(j)>1:
                    search = j[1]+":"+str(df['pmid'][i])
                    j = j[0]
                    #print(j)
                    #print(search)

                    if s != 'None':
                        if str(j) in geneid_genename_goterm.keys():
                            geneid_genename_goterm[str(j)] = str(geneid_genename_goterm[str(j)]) + "##" + s
                        else:
                            geneid_genename_goterm[str(j)] = s

                    geneids.append(str(j))
                    geneid_iffunction[str(j)] = 0
                    if str(j) in geneid_functions.keys():
                        geneid_iffunction[str(j)] = 1
                    if str(j) in geneid_pmid.keys():
                        geneid_pmid[str(j)] = str(geneid_pmid[str(j)])+"##"+str(df['pmid'][i])
                    else:
                        geneid_pmid[str(j)] = str(df['pmid'][i])
                    if str(j) in geneid_gene.keys():
                        geneid_gene[str(j)] = str(geneid_gene[str(j)])+"##"+str(df['pmid'][i])+":"+str(df['gene'][i])+"("+search+")"
                    else:
                        geneid_gene[str(j)] = str(df['pmid'][i])+":"+str(df['gene'][i])+"("+search+")"
                    if str(j) in geneid_species.keys():
                        geneid_species[str(j)] = str(geneid_species[str(j)])+"##"+str(df['pmid'][i])+":"+str(df['species'][i])+"("+search+")"
                    else:
                        geneid_species[str(j)] = str(df['pmid'][i])+":"+str(df['species'][i])+"("+search+")"
                else:
                    print("error ",j)
    #print(len(list(set(df['gene']))))
    #df = pd.read_csv("entity_link_2_summery_mb_entity_name_bert_2.csv")
    df_1 = pd.read_csv(geneid_gene_specie_withfunction_file,sep='\t')
    functions = ['methyltransferase activity','phosphatase activity','guanylyltransferase activity',"RNA capping",'RNA methylation','pyrophosphatase activity','S-adenosylmethionine-dependent methyltransferase activity',"O-methyltransferase activity","N-methyltransferase activity","7-methylguanosine RNA capping"]

    df_go = pd.read_csv("go_mb.csv")
    go = df_go['Name'].tolist()
    go_id = df_go['ID'].tolist()
    id_gene_dict = {}
    id_species_dict = {}
    functions_id = []
    df = pd.read_csv(geneid_gene_specie_withfunction_file,sep='\t')
    num = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(functions)):
        functions_id.append(go_id[go.index(str(functions[i]))])
    df_go = pd.read_csv(geneid_gene_specie_withfunction_file,sep='\t')
    go_id = df_go['ID'].tolist()
    print(functions_id)
    #print(geneid_genename_goterm['462226'])
    '''
    ns = 0
    print("loading result ", i / len(df_function))
    for i in geneid_genename_goterm.keys():
        ns += 1
        if ns % len(df_function) / 10 == 0:
            
        goids = geneid_genename_goterm[i]
        parents = []
        for j in goids.split("##"):
            parents.append(j)
            if GO.query_term(str(j)):
                p = GO.query_term(str(j)).get_all_parents()
                for k in p:
                    parents.append(str(k))
        parents = list(set(parents))
        for s in functions_id:
            if str(s) in parents:
                num[functions_id.index(s)].append(str(i))
    

    n = []
    t = 0
    
    for i in num:
        print(functions[t])
        t+=1
        for j in list(set(i)):
            print(j[0]," -- ",j[1])
    

    for i in num:
        n.append(list(set(i)))
        
    for j in n[0]:
            if j in n[1] and j in n[2]:
                #for s in gene_candidates:
                 #   if s in str(id_gene_dict[str(j)]).lower():
                        print(id_gene_dict[str(j)],id_species_dict[str(j)])
                      #  break
    '''
    geneids = list(set(geneids))
    #print(len(geneids))
    #print(geneids[:100])
    functions = []
    genes = []
    species = []
    pmids = []
    gcns = []
    isfunction = []
    methy = []
    guan = []
    pyro = []
    phos = []
    rna_methy = []
    r_capping = []
    function_entitys = []
    tick = 0
    for i in geneids:
        tick+=1
        if tick % len(geneid) / 10 == 0:
            print("final integrate result ", tick / len(geneid))
        if str(i) in geneid_function_entitys.keys():
            function_entitys.append(geneid_function_entitys[str(i)])
            isfunction.append(1)
        else:
            isfunction.append(0)
            function_entitys.append("None")
        if str(i) in geneid_functions.keys():
            functions.append(geneid_functions[str(i)])
        else:
            functions.append("None")
        if str(i) in geneid_gene.keys():
            genes.append(geneid_gene[str(i)])
        else:
            genes.append("None")
        if str(i) in geneid_species.keys():
            species.append(geneid_species[str(i)])
        else:
            species.append("None")
        if str(i) in geneid_pmid.keys():
            pmids.append(geneid_pmid[str(i)])
        else:
            pmids.append("None")

    df = pd.DataFrame({"gene_id":geneids,"pmid":pmids,"isfunctioned":isfunction,"function_entitys":function_entitys,"function_linked":functions,"genes":genes,"species":species})
    #df.to_csv("final_siderophores_geneid_02_24.csv")
    df.to_csv(save_file,sep='\t')
    print("thanks for using")
