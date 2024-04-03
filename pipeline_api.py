#! python
# -*- coding=utf-8 -*-

# ******************************************************************************
#               > File Name: entrez.py
#               > Author: JamesWang(Wang Zimo)
#               > Mail: JamesWangZimo2020@gmail.com
#               > Created Time: 2020年07月27日 星期一 13时13分02秒
# ******************************************************************************
import multiprocessing as mp
import re
from urllib.error import URLError, HTTPError
from retrying import retry
import time
import pandas as pd
import pysnooper
from Bio import Entrez
from Bio.Entrez.Parser import ValidationError
import sqlite3
import traceback
from multiprocessing import RLock
import ssl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
ssl._create_default_https_context = ssl._create_unverified_context
Entrez.email = '1847156239@qq.com'
Entrez.api_key = '73664c0b675e8450f3cf90add93e69820808'

ranks_over_species = ('class', 'family', 'subclass', 'genus', 'tribe',
                      'subtribe', 'parvorder', 'suborder', 'infraorder',
                      'subgenus', 'superorder', 'superfamily', 'order',
                      'section', 'subfamily', 'subsection', 'series',
                      'infraclass', 'species subgroup', 'subcohort',
                      'species group', 'cohort')
ranks_under_species = ('genotype', 'subvariety', 'morph', 'subspecies',
                       'forma specialis', 'pathogroup', 'varietas', 'biotype',
                       'serogroup', 'isolate', 'strain', 'forma', 'serotype')

# buffers
sp_dict = {}
pair_buffer = {}

too_much = float('Inf')


def search_gene_species(pairs, i, lock, summary=False, short=True,
                        dic_max=10_000_000, parse_pairs=True):
    print(f'search_gene_species {i}')
    del i
    error_records = []
    '''
    if isinstance(db, str):
        con = sqlite3.connect(db)
        db = con.cursor()
    # initiating buffer
    global sp_dict, pair_buffer

    if len(sp_dict) > dic_max:
        sp_dict = {}

    if len(pair_buffer) > dic_max:
        pair_buffer = {}

    # spliting by dictionary result
    '''
    if parse_pairs:
        for i, pair in enumerate(pairs):
            pairs[i] = get_dict_res_pair(pair)

    # find species

    # print(len(pairs))
    for pair in pairs:
        sp = pair[0][0]

        if sp in sp_dict:
            # print('continue')
            continue
        # print(pair[0])
        # spid, dblen, dbexlen, dbexsp, etzlen = found_species(pair[0], db, lock)
        # print(spid)
        # sp_dict[sp] = spid

    # print(sp_dict)
    # print(len(pairs), pairs)
    # find genes
    found_pairs = {}

    for p, pair in enumerate(pairs):
        sp = pair[0]
        gene = pair[1]
        gene_names = pair[1]
        # print("gene",pair)
        pair = (pair[0], pair[1], pair[2])
        # print(pair)
        '''
        if len(sp_dict[sp]) == 0:
            continue

        if too_much in sp_dict[sp]:
            e = {'pair': pairs[p]}
            e['type'] = 'too many species'
            error_records.append(e)
            continue
        '''
        '''
        sp_ids = sp_dict[sp] - {too_much}
        '''
        sp_ids = pair[2]
        if pair in pair_buffer:
            if pair_buffer[pair] == {too_much}:
                continue
            found_pairs.update(pair_buffer[pair])

        # gene_names = sorted(set(gene_names), key=gene_names.index)
        gene_names = str(gene_names).split(" ")
        print('gene names:', gene_names)
        genes = {}
        tax_ids = pair[2]
        '''
        for gene_name in gene_names:
            tax_ids = tuple({tax_entry[0]
                                 for tax_entry in sp_ids})
            ukbids = set()
            tax_sql = ' or '.join(('tax_id=?', ) * len(tax_ids))
            tax_ids = tuple(tax_ids)

            genes_uniprot = {}
        '''
        print('dbsearch:', genes)
        if len(genes) > 100:
            pair_buffer[(sp, gene)] = {too_much}
            e = {'pair': pairs[p]}
            e['type'] = 'too many genes'
            error_records.append(e)
            continue
        print('genes:', genes)
        if not genes:
            for i, gene_name in enumerate([gene_names]):

                print(171, 'lock',
                      time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(time.time())))
                lock.acquire()
                http_error = False
                try:
                    record = esearch_gene(gene_name[0],
                                          taxids=tax_ids,
                                          summary=summary)
                    print("record ", record)
                except (HTTPError, ValidationError):
                    e = {'pair': pairs[p]}
                    e['type'] = 'http error'
                    error_records.append(e)
                    http_error = True
                print(185, 'release',
                      time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(time.time())))
                lock.release()
                if http_error:
                    continue

                genes_ = {smr.attributes['uid']: dict(smr) for smr in
                          record['Summary']}
                genes_.update({uid: {} for uid in record['IdList']
                               if uid not in genes_})
                # if '116889802' in record['IdList']:
                #     print(record)
                #     print(genes_)
                # print('genes:', genes)

                for uid in genes_:
                    genes_[uid]['find'] = 'esearch'
                    genes_[uid]['pair'] = pair
                    genes_[uid]['species_info'] = sp_ids
                    genes_[uid]['gene_name'] = gene_name

                genes.update(genes_)
                if i == 0 and len(genes_) != 0:
                    break

            if too_much in genes:
                del genes[too_much]
            elif -too_much in genes:
                del genes[-too_much]

            if summary and short:
                for uid in genes:
                    if 'Description' not in genes[uid]:
                        # print('Error: no description', uid, genes[uid].keys())
                        print(220, 'lock',
                              time.strftime('%Y-%m-%d %H:%M:%S',
                                            time.localtime(time.time())))
                        lock.acquire()
                        http_error = False
                        try:
                            genes[uid].update(new_esummary(
                                db='gene', id=uid)[0])
                        except (HTTPError, ValidationError):
                            e = {'pair': pairs[p]}
                            e['type'] = 'http error'
                            error_records.append(e)
                            http_error = True

                        print(234, 'release',
                              time.strftime('%Y-%m-%d %H:%M:%S',
                                            time.localtime(time.time())))
                        lock.release()

                        if http_error:
                            continue

                    genes[uid] = {'find': genes[uid]['find'],
                                  'pair': genes[uid]['pair'],
                                  'species_info':
                                      genes[uid]['species_info'],
                                  'gene_name': genes[uid]['gene_name'],
                                  'Description':
                                      str(genes[uid]['Description']),
                                  'GenomicInfo':
                                      [{k: str(g[k]) for k in g}
                                       for g in genes[uid]['GenomicInfo']],
                                  'Name': str(genes[uid]['Name']),
                                  'NomenclatureName':
                                      str(genes[uid]['NomenclatureName']),
                                  'NomenclatureStatus':
                                      str(genes[uid]['NomenclatureStatus']),
                                  'NomenclatureSymbol':
                                      str(genes[uid]['NomenclatureSymbol']),
                                  'Organism':
                                      str(genes[uid]['Organism'] \
                                              ['ScientificName']),
                                  'tax_id':
                                      str(genes[uid]['Organism']['TaxID']),
                                  'OtherAliases':
                                      str(genes[uid]['OtherAliases']),
                                  'OtherDesignations':
                                      str(genes[uid]['OtherDesignations'])}

                    if len(genes[uid]['GenomicInfo']) != 0:
                        genes[uid]['GenomicInfo'] = \
                            [dict(genomic_info) for genomic_info in
                             genes[uid]['GenomicInfo']]
                    else:
                        genes[uid]['GenomicInfo'] = []

                    for ginfo in genes[uid]['GenomicInfo']:
                        for ginfo_key in ginfo:
                            ginfo[ginfo_key] = \
                                str(ginfo[ginfo_key])

        pair_buffer[(sp, gene)] = genes

        found_pairs.update(genes)
        # found_pairs.update(genes_uniprot)

    return found_pairs, error_records


def get_overlapped_res(ind_s):
    '''
    Given an indexed_str pair, return their's overlapped indexed_str

    this fundtion works for entities in deduplicated pair
    '''

    if 'overlapped' in ind_s.note:
        return [o for en in ind_s.note['overlapped'] for o in en]

    return None


def get_dict_res_en(en):
    '''
    Given an entity, return itself and dictionary map result of this entity
    '''

    if en.note['EnExWay'] == 'both':
        ens = get_overlapped_res(en)

        if ens is not None:
            ens = tuple(e_ for e_ in ens
                        if e_.note['EnExWay'] in ('acronym', 'dict'))  # e for error, e_ for entity
    elif en.note['EnExWay'] == 'acronym':
        ens = (en.note['source'],)
    elif en.note['EnExWay'] == 'bern':
        ens = ()
    elif en.note['EnExWay'] == 'dict':
        ens = ()

    ens = (en, *ens)
    ens = tuple(sorted(set(ens), key=ens.index))

    return ens


def get_dict_res_pair(pair):
    '''
    Given a pair, return dictionary map result of this pair
    '''
    species = pair[0]
    gene = pair[1]

    species = get_dict_res_en(species)
    gene = get_dict_res_en(gene)

    return (species, gene)


def found_species(words, db, lock):
    tax = set()
    # print(words)

    for word in words:
        db.execute('''select tax_id, rank from names
                   indexed by name_index_names
                   where name=?''', (word,))
        tax.update({t for t in map(lambda _: (*_, 'dbsearch', word),
                                   db.fetchall())
                    if t[1].lower() != 'no rank'})
    dblen = len(tax)
    dbexlen = len(tuple(t for t in tax if t[1].lower() in ranks_over_species))
    # print(tax)
    has_rank_over_species = 'species' not in {t[1].lower() for t in tax}

    while has_rank_over_species:
        has_rank_over_species = False
        tax_sp = {t for t in tax if t[1].lower() == 'species'}
        tax_over = tax - tax_sp

        for it in tax_over:
            tax.remove(it)
            db.execute('''select tax_id, rank from nodes
                       indexed by parent_id_index_nodes
                       where parent_id=?''', (it[0],))
            tax.update({(*t, 'dbexpand', it) for t in db.fetchall()})

        for t in tax:
            if t[1].lower() in ranks_over_species:
                has_rank_over_species = True
                break

    dbexsp = len(tax) - dblen + dbexlen
    if len(tax) > 100:
        tax = {too_much}
    # print(tax)

    if len(tax) == 0:
        # try:
        tax = {}
        print(380, 'lock',
              time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time())))
        lock.acquire()
        for word in words:
            try:
                tax.update({word: esearch_species(word)['IdList']})
            except (HTTPError, ValidationError):
                tax.update({word: [too_much]})
        print(389, 'release',
              time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time())))
        lock.release()
        # except URLError as error:
        #     traceback.print_exc(error)
        #     print(error)
        #
        #     return (set(), 0, 0, 0, 0)

        if len(tax) == 0:
            return (set(), 0, 0, 0, 0)

        new_tax = set()
        for word in words:
            if len(tax[word]) == 0:
                continue
            if tax[word] == [too_much]:
                new_tax.update({too_much})
                continue
            tax_ids = tax[word]
            where_str = ' or '.join(('tax_id=?',) * len(tax_ids))
            db.execute(f'''select tax_id, rank from names
                       indexed by tax_id_index_names
                       where ({where_str})''', tuple(tax_ids))
            new_tax.update({t for t in set(map(lambda _: (*_, 'esearch', word),
                                               db.fetchall()))
                            if t[1] != 'no rank'})
        tax = new_tax
        etzlen = len(tax)
    else:
        etzlen = 0
    # print(tax)

    return (tax, dblen, dbexlen, dbexsp, etzlen)


def new_idlist(record, db_ncbi, p_retstart=None, p_retmax=None):
    handle = Entrez.efetch(db=db_ncbi,
                           WebEnv=record['WebEnv'],
                           query_key=record['QueryKey'],
                           retmax=p_retmax, retstart=p_retstart,
                           rettype='uilist', retmode='text')
    record['IdList'] += list(map(lambda _: _.strip(),
                                 handle.readlines()))


def print_error(e):
    print('Error!', type(e), e)
    return True


@retry(wait_fixed=2000, stop_max_attempt_number=10,
       retry_on_exception=print_error)
def new_esummary(**kwargs):
    handle = Entrez.esummary(**kwargs)
    summary = Entrez.read(handle)
    if 'DocumentSummarySet' in summary:
        s = summary['DocumentSummarySet']['DocumentSummary']
    return s


#  @pysnooper.snoop()

@retry(wait_fixed=2000, stop_max_attempt_number=10,
       retry_on_exception=print_error)
def esearch_gene(query, taxids=None, species=None, summary=False):
    print('query:', query, 'taxids:', taxids)
    records = []
    if taxids == None:
        if species == None:
            raise ValueError('taxid or species scientific name needed')

        for sp in species:
            records.append(
                Entrez.read(
                    Entrez.esearch(db="gene",
                                   term=f"({query}[All Fields] AND '\
                                   '{sp}[All Fields]) AND alive[prop]",
                                   usehistory='y')))
    else:
        if len(taxids) == 1:
            taxids_str = f'txid{taxids[0]}[All Fields]'
        else:
            taxids_str = [f'txid{taxid}[All Fields]' for taxid in taxids]
            taxids_str = ' OR '.join(taxids_str)
            taxids_str = f'({taxids_str})'
        # print('taxids_str', taxids_str)

        # print(f"({query}[All Fields]"\
        #                        f" AND {taxids_str}) AND alive[prop]")
        record = Entrez.read(
            Entrez.esearch(db="gene",
                           term=f"({query}[All Fields]" \
                                f" AND {taxids_str}) AND alive[prop]",
                           usehistory='y'))

    print('records ids found:', sum([int(r['Count']) for r in records]))
    record_ret = {}

    # print('record:', record)
    count = int(record['Count'])
    record_ret['Count'] = count
    # print('record_ret:', record_ret)

    if count > 20:
        if count > 100:
            record['IdList'] = [too_much]

            # for i in range(count // 500):
            #     new_idlist(record, 'gene', i*500, 500)
            # new_idlist(record, 'gene', (i+1)*500)
        else:
            new_idlist(record, 'gene')

    del record['RetMax'], record['RetStart']
    record_ret['IdList'] = record['IdList']
    # print('record_ret:', record_ret)

    if summary:
        if 0 < count <= 100:
            # if '116889802' in record['IdList']:
            #     print(query)
            #     print(taxids)
            #     record['IdList'].index('116889802')
            #     print(count)
            record['Summary'] = new_esummary(db='gene',
                                             WebEnv=record['WebEnv'],
                                             query_key=record['QueryKey'])

            # if '116889802' in record['IdList']:
            #    print(record)
        else:
            record['Summary'] = {}
    else:
        record['Summary'] = {}

    record_ret['Summary'] = record['Summary']
    # print('record_ret:', record_ret)

    # print('record_ret:', record_ret)
    # if '116889802' in record_ret['IdList']:
    #    print(record_ret)
    return record_ret


@retry(wait_fixed=2000, stop_max_attempt_number=10,
       retry_on_exception=print_error)
def esearch_species(query):
    handle = Entrez.esearch(db='taxonomy',
                            term='%s[All Names]' % query,
                            usehistory='y')
    record = Entrez.read(handle)
    # print('esearch species all names:', record['Count'])

    if record['Count'] == '0':
        handle = Entrez.esearch(db='taxonomy',
                                term='%s[Name Tokens]' % query,
                                usehistory='y')
        record = Entrez.read(handle)

    count = int(record['Count'])
    # print('esearch species name tokens:', record['Count'])

    if count > 20:
        if count > 100:
            record['IdList'] = [too_much]

            # for i in range(count // 500):
            #     new_idlist(record, 'taxonomy', i*500, 500)
            # new_idlist(record, 'taxonomy', (i+1)*500)
        else:
            new_idlist(record, 'taxonomy')

    del record['RetMax'], record['RetStart']

    return record


def download_gene_fasta(path, gene_ids=None, query_key=None, WebEnv=None):
    if gene_ids == None:
        if query_key == None or WebEnv == None:
            raise RuntimeError('Empty id list - nothing todo')

    handle = Entrez.elink(db='nucleotide', dbfrom='gene', id=gene_ids,
                          query_key=query_key, WebEnv=WebEnv)
    records = Entrez.read(handle)

    for record in records:
        download_nucleotide_ids = {l['Id']
                                   for link in record['LinkSetDb']

                                   for l in link['Link']}
        gb = Entrez.efetch(db="nucleotide",
                           id=download_nucleotide_ids,
                           rettype='gb',
                           retmode='text').read()

        for gene_id in record['IdList']:
            with open(f"{path}/{gene_id}.gb", 'w') as handle:
                handle.write(gb)


def read_valid_pair_to_id(handle):
    if isinstance(handle, str):
        pass
    elif str(type(handle)) == "<class '_io.TextIOWrapper'>":
        pass
    else:
        raise ValueError('file path needed')
    df = pd.read_csv(handle, sep='\t',
                     names=('species', 'gene', 'gene_id',
                            'newentry_id', 'nucleotide_id'))

    return df


def read_result():
    with open('logs/pair_to_seq_long') as handle:
        ret = []
        setstr = []

        for line in handle:
            if re.match(r'[0-9]{30}', line):
                if setstr:
                    ret.append(eval(''.join(setstr)))
                setstr = []

                continue
            line = line.strip()

            if line.endswith("'") or line.endswith('"'):
                line += '+\\' + '\n'
            else:
                line += '\n'
            setstr.append(line)

    return ret


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


# set_gene_species_pairs = func(set_gene_species_pairs,2179)
from multiprocessing import Manager, Pool
def geneid_concat(num,save_file_string,save_file):
    gene = []
    species = []
    pmid = []
    gene_id = []
    for i in range(num):
        df = pd.read_csv(save_file_string + str(i) + ".tsv", sep='\t')
        for i in range(len(df)):
            gene.append(df['gene'][i])
            species.append(df['species'][i])
            pmid.append(df['pmid'][i])
            gene_id.append(df['gene_ids'][i])
    df = pd.DataFrame({'pmid': pmid, 'gene': gene, 'species': species, 'gene_id': gene_id})
    df.to_csv(save_file,sep='\t')

def multi_search_and_save_protein(gene_species_file,num,save_file_string):
    df = pd.read_csv(gene_species_file,sep='\t')
    pmids = []
    genes = []
    species = []
    taxids = []
    geneids = []
    total = 0
    temp_length = int(len(df)/8)+1
    try:
        dfs = pd.read_csv(save_file_string + str(num) + ".tsv",sep='\t')
        pmids = dfs["pmid"]
        genes = dfs["gene"]
        species = dfs["species"]
        taxids = dfs['taxids']
        geneids = dfs['protein_ids']
        bias = len(dfs)
    except:
        bias = 0
    for i in range(bias+num * temp_length, min(num * temp_length + temp_length, len(df))):
        total += 1
        inds = {}
        if df['gene'][i] != 'None' and df['species'][i] != 'None':
            sp = str(df['species'][i])
            gene = str(df['gene'][i])
            for s in sp.split(" "):
                gene = gene.replace(s,"")

            print(num, " ", total)
            pmids.append(str(df['pmid'][i]))
            genes.append(str(df['gene'][i]))
            species.append(str(df['species'][i]))
            temp = []
            r = []
            for trys in range(1):
                records = esearch_species(str(df["species"][i]))
                #print("thread ", str(num), " trys_species ", trys, " ", records['IdList'])
                for j in records['IdList']:
                    r.append(str(j))
            if len(r) > 0:
                taxids.append("##".join(r))
            else:
                taxids.append("None")
            #all search
            for sps in sp.split("##"):
                record_new = Entrez.read(
                                Entrez.esearch(db="proteins",
                                               term=f"({str(gene)}[All Fields]" \
                                                    f" AND {str(sps)}) AND alive[prop]",
                                               usehistory='y',sort = 'relevance'))
                #print("thread ", str(num),  " ", record_new['IdList'])
                for s in record_new['IdList']:
                    #print(record_new['IdList'].index(s))
                    temp.append(s)
                    if s in inds.keys():
                        inds[s] = inds[s] + '|' + gene + "-" + sps + '-' + str(record_new['IdList'].index(s))
                    else:
                        inds[s] = gene + "-" + sps + '-' + str(record_new['IdList'].index(s))
            temp = list(set(temp))
            temp_all = []
            for s in temp:
                temp_all.append(s+'('+inds[s]+')')
            print(num, "genes ", temp_all)
            if len(temp) > 0:
                geneids.append("##".join(temp_all))
            else:
                geneids.append("None")



        else:
            pmids.append(str(df['pmid'][i]))
            genes.append(str(df['gene'][i]))
            species.append(str(df['species'][i]))
            taxids.append("None")
            geneids.append("None")
        dfs = pd.DataFrame(
                {"pmid": pmids, "gene": genes, "species": species, 'taxids': taxids, 'protein_ids': geneids})
        dfs.to_csv(save_file_string + str(num) + ".tsv",sep='\t')


def multi_search_and_save(gene_species_file,num,save_file_string):
    df = pd.read_csv(gene_species_file,sep='\t')
    pmids = []
    genes = []
    species = []
    taxids = []
    geneids = []
    total = 0
    temp_length = int(len(df)/8)+1

    try:
        dfs = pd.read_csv(save_file_string + str(num) + ".tsv",sep='\t')
        pmids = dfs["pmid"].tolist()
        genes = dfs["gene"].tolist()
        species = dfs["species"].tolist()
        taxids = dfs['taxids'].tolist()
        geneids = dfs['gene_ids'].tolist()
        bias = len(dfs)
    except:
        bias = 0

    for i in range(bias+(num * temp_length), min(num * temp_length + temp_length, len(df))):
        total += 1
        inds = {}
        if df['gene'][i] != 'None' and df['species'][i] != 'None':
            sp = str(df['species'][i])
            gene = str(df['gene'][i])
            for s in sp.split(" "):
                gene = gene.replace(s,"")

            print(num, " ", bias+total)
            pmids.append(str(df['pmid'][i]))
            genes.append(str(df['gene'][i]))
            species.append(str(df['species'][i]))
            temp = []
            r = []
            taxids.append("no_search")
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
                        print("thread ", str(num),  " ",str(gene)," ", record_new['IdList'])
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
                print(num, "genes ", temp_all)
                if len(temp) > 0:
                    geneids.append("##".join(temp_all))
                else:
                    geneids.append("None")



        else:
            pmids.append(str(df['pmid'][i]))
            genes.append(str(df['gene'][i]))
            species.append(str(df['species'][i]))
            taxids.append("None")
            geneids.append("None")
        dfs = pd.DataFrame(
                {"pmid": pmids, "gene": genes, "species": species, 'taxids': taxids, 'gene_ids': geneids})
        dfs.to_csv(save_file_string + str(num) + ".tsv",sep='\t')

def main():
    query = 'PfEMP1'
    j = 'falciparum malaria'
    record_new = Entrez.read(
        Entrez.esearch(db="gene",
                       term=f"({query}[All Fields]" \
                            f" AND {j})",
                       usehistory='y',sort = 'relevance'))#,RetMax = '900'
    print("test ", record_new)
    for i in record_new['IdList']:
        if '22' in i:
            print(i)
    pool = mp.Pool(8)
    ps = []
'''
    for i in range(8):
        ps.append(mp.Process(target=multi_search_and_save,args=(i,)))
    for p in ps:
        p.start()
'''
'''
'''
if __name__ == '__main__':
    main()
