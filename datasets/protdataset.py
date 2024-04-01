import torch

class ProtSeqDETRDataset(torch.utils.data.Dataset):
    def __init__(self, id_ec, ec_id, id_seq, max_labes, esm_layer=32, ec_to_label=None, label_to_ec=None):
        super().__init__()

        self.id_list = list(id_ec.keys())
        self.ec_list = list(ec_id.keys())
        self.seq_list = list(id_seq.values())
        
        if ec_to_label is None:
            self.num_labels = len(self.ec_list)
            self.ec_to_label = {ec: i for i, ec in enumerate(self.ec_list)}
            self.label_to_ec = {v: k for k, v in self.ec_to_label.items()}
        else:
            self.num_labels = len(ec_to_label)
            self.ec_to_label = ec_to_label
            self.label_to_ec = label_to_ec

        self.labels = []  
        self.onehot_labels = []  
        self.feat = []  
        self.ec_weight = torch.zeros(self.num_labels + 1)  # the last is for no ec
        
        for id in self.id_list:
            ecs = id_ec[id]
            label = []
            onthot_label = torch.zeros(self.num_labels + 1)  # the last is for no ec
            for ec in ecs:
                if ec in self.ec_to_label:
                    label_index = self.ec_to_label[ec]
                    self.ec_weight[label_index] += 1
                    label.append(label_index)
                    onthot_label[label_index] = 1.0
            # label to tensor
            self.labels.append(torch.tensor(label))
            self.onehot_labels.append(onthot_label)

        self.ec_weight[-1] = len(self.id_list) * max_labes - self.ec_weight.sum()

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id = self.id_list[idx]
        label = self.labels[idx]
        onehot_label = self.onehot_labels[idx]
        seq = self.seq_list[idx]
        return id, label, onehot_label, seq

    def collate_fn(self, batch):
        ids, labels, onehot_labels, seqs = zip(*batch)
        onehot_labels = torch.stack(onehot_labels)
        return ids, seqs, labels, onehot_labels