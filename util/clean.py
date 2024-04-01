import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle
import time
from tqdm import tqdm
import pdb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])


    for ec in ec_id.keys():
        ec_id[ec] = sorted(ec_id[ec])
        
    return id_ec, ec_id

def get_id_seq_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_seq = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_seq[rows[0]] = rows[2].strip()
    return id_seq
    
def add_mem(id_ec, ec_id, mem):
    id_ec_mem = id_ec.copy()
    ec_id_mem = ec_id.copy()

    id_ec_mem.update(mem) 
    for id in mem.keys():
        for ec in mem[id]:
            if ec not in ec_id_mem.keys(): 
                ec_id_mem[ec] = set()
                ec_id_mem[ec].add(id)
            else:
                ec_id_mem[ec].add(id)

    return id_ec_mem, ec_id_mem

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        if 'mean_representations' in a:
            if int(32) in a['mean_representations']:
                emb = a['mean_representations'][32]
            elif int(33) in a['mean_representations']:
                emb = a['mean_representations'][33]

        elif 'representations' in a:
            # emb = a['representations'][33]
            if int(32) in a['representations']:
                emb = a['representations'][32]
            elif int(33) in a['representations']:
                emb = a['representations'][33]
        else:
            raise KeyError
    return emb


def load_esm(lookup, type="clean"):
    base_dir = "./data/esm_data" if type == "clean" else "./ecrecer_data/esm_data"
    file_path = os.path.join(base_dir, f'{lookup}.pt')
    
    esm = format_esm(torch.load(file_path))
    return esm.unsqueeze(0)


def esm_embedding(ec_id_dict, device, dtype, type="clean"):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    for ec in tqdm(list(ec_id_dict.keys())):
    # for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id, type=type) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(id_ec_test, model, device, dtype, data_type="clean"):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id, type=data_type) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb

def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_name, layer=32):
    esm_script = "esm/scripts/extract.py"
    esm_out = f"embs/esm{layer}" 
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "data/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean", "--repr_layers", str(layer)]
    subprocess.run(command)

# TODO not implemented yet
def compute_esm(train_file):
    _, ec_id = get_ec_id_dict('./data/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    
def prepare_infer_fasta(fasta_name):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open('./data/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('./data/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name, r=10, type="clean") :
    csv_file = open('./data/'+ csv_name + '.csv') if type=="clean" else open('./ecrecer_data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/' + fasta_name + '.fasta','w') if type=="clean" else open('./ecrecer_data/' + fasta_name + '.fasta','w')
    single_id = set(single_id)
    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(r):
                seq = rows[2].strip()
                mu, sigma = .10, .02 # mean and standard deviation
                s = np.random.normal(mu, sigma, 1)
                mut_rate = s[0]
                times = math.ceil(len(seq) * mut_rate)
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                seq = seq.replace('*', '<mask>')
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def is_mutated(id, r=10, type="clean"):
    base_dir = "./data/esm_data" if type == "clean" else "./ecrecer_data/esm_data"

    for i in range(r):
        file_path = os.path.join(base_dir, f'{id}_{i}.pt')
        if not os.path.exists(file_path):
            return False
    return True

def mutate_single_seq_ECs(train_file, type="clean"):
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv') if type=="clean" else get_ec_id_dict('./ecrecer_data/' + train_file + '.csv')

    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    single_id = set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            # if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
            if ec in single_ec and not is_mutated(id, type=type):
                single_id.add(id)
                break
    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    mask_sequences(single_id, train_file, train_file+'_single_seq_ECs', type=type)
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name

def mutate_all_seq_ECs(train_file, type="clean"):
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv') if type=="clean" else get_ec_id_dict('./ecrecer_data/' + train_file + '.csv')
    
    id_to_mutated = set()
    for id in id_ec.keys():
        if not is_mutated(id, 1, type=type):
            id_to_mutated.add(id)

    print("Number of total sequences:",len(id_ec.keys()))
    print("Number of sequences need to mutate: ",len(id_to_mutated))
    print("Number of sequences already mutated: ", len(id_ec.keys()) - len(id_to_mutated))
    mask_sequences(id_to_mutated, train_file, train_file+'_all_seq_ECs', r=1, type=type)
    fasta_name = train_file+'_all_seq_ECs'
    return fasta_name

def combine_csv(mem_pth, session_pth, combined_pth):
    
    with open(combined_pth, 'w') as output_file:
        with open(mem_pth, 'r') as file1:
            output_file.write(file1.read())
        
        with open(session_pth, 'r') as file2:
            next(file2) 
            output_file.write(file2.read())

def get_true_labels(file_name):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label


def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label

def get_eval_metrics(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    # print(label_pos_dict)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels = pred_label[i]
        # for label, prob in zip(labels, probs):
        #     if label in all_label:
        #         pos = label_pos_dict[label]
        #         pred_m_auc[i, pos] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    # roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, acc