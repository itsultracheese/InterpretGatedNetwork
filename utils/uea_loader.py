import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from aeon.datasets import load_from_tsfile
from aeon.transformations.collection.interpolate import TSInterpolator


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standard', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standard":
            if self.mean is None:
                self.mean = np.mean(x, axis=-1, keepdims=True)
                self.std = np.std(x, axis=-1, keepdims=True)
            return (x - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = np.max(x, axis=-1, keepdims=True)
                self.min_val = np.min(x, axis=-1, keepdims=True)
            return (x - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))
        

class UEADataset(Dataset):
    def __init__(
            self,
            dataset,
            root_dir='./data/UEA_multivariate',
            flag="TRAIN",
            normalizer=None,
            label_encoder=None
        ):
            self.file_path=f"{root_dir}/{dataset}/{dataset}_{flag}.ts"
            self.flag = flag

            self.normalizer = Normalizer() if normalizer is None else normalizer
            self.label_encoder = LabelEncoder() if label_encoder is None else label_encoder
            self.fit = label_encoder is None

            self.x, self.y = self.load()
            self.num_class = np.unique(self.y).shape[0]

    def load(self):
        data_x, data_y = load_from_tsfile(self.file_path)

        # Interpolate into same length
        max_len = max([x.shape[1] for x in data_x])
        interpolator = TSInterpolator(max_len)
        data_x = interpolator.fit_transform(data_x)

        # Normalize
        data_x = self.normalizer.normalize(data_x)
        if self.fit:
            data_y = self.label_encoder.fit_transform(data_y)
        else:
            data_y = self.label_encoder.transform(data_y)

        return data_x, data_y
    
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(self.y[[index]]).long()
    
    def __len__(self):
        return self.x.shape[0]
    

# def load_UCR_data(dataset_name,
#                   dataset_dir,
#                   from_pt=False,
#                   save_pt=False,
#                   normalize_data=False,
#                   center=False,
#                   scale=False,
#                   batch_size=256,
#                   no_val=False,
#                   return_encoder=False,
#                   print_dataset_info=False,
#                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
#     if print_dataset_info:
#         print("="*5)
#         print("[{}]".format(dataset_name))
#         print(dataset_dir)
    
#     # Load from compiled .pt files.
#     if from_pt:
#         try:
#             if normalize_data:
#                 train_path = dataset_dir + os.sep + dataset_name + "_TRAIN_N.pt"
#                 test_path = dataset_dir + os.sep + dataset_name + "_TEST_N.pt"
#                 val_path = dataset_dir + os.sep + dataset_name + "_VAL_N.pt"
#             else:
#                 train_path = dataset_dir + os.sep + dataset_name + "_TRAIN.pt"
#                 test_path = dataset_dir + os.sep + dataset_name + "_TEST.pt"
#                 val_path = dataset_dir + os.sep + dataset_name + "_VAL.pt"

#             train_dataset = torch.load(train_path)
#             test_dataset = torch.load(test_path)
#             val_dataset = torch.load(val_path)

#             if print_dataset_info:
#                 print("Loaded data from .pt file")
            
#         except:
#             print("Does not find .pt file.")
#             from_pt = False
    
#     # Load from raw .ts files.
#     if not from_pt:
        
#         # Read data
#         train_path = dataset_dir + os.sep + dataset_name + "_TRAIN.tsv"
#         test_path = dataset_dir + os.sep + dataset_name + "_TEST.tsv"

#         if print_dataset_info:
#             print("Loading data from .tsv file")
        
#         train_data = pd.read_csv(train_path, delimiter='\t', header=None).values
#         test_data = pd.read_csv(test_path, delimiter='\t', header=None).values
        
#         train_x, train_y = np.expand_dims(train_data[:, 1:], axis=1), train_data[:, 0]
#         test_x, test_y = np.expand_dims(test_data[:, 1:], axis=1), test_data[:, 0]
        
#         # Split training & validtion sets
#         if train_x.shape[0] < 100 or no_val:
#             val_x, val_y = test_x.copy(), test_y.copy()
#         else:
#             # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
#             train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
        
#         data_preprocess = DataPreprocess(train_x, train_y, test_x, test_y, val_x, val_y)
#         train_x, train_y, test_x, test_y, val_x, val_y = data_preprocess.transform(normalize=normalize_data)

#         train_x_tensor = torch.tensor(train_x, device=device).float()
#         train_y_tensor = torch.tensor(train_y, device=device).long()
#         test_x_tensor = torch.tensor(test_x, device=device).float()
#         test_y_tensor = torch.tensor(test_y, device=device).long()
#         val_x_tensor = torch.tensor(val_x, device=device).float()
#         val_y_tensor = torch.tensor(val_y, device=device).long()
        
#         train_dataset = TscDataset(train_x_tensor, train_y_tensor)
#         test_dataset = TscDataset(test_x_tensor, test_y_tensor)
#         val_dataset = TscDataset(val_x_tensor, val_y_tensor)
        
#         if save_pt:
#             if normalize_data:
#                 torch.save(train_dataset, dataset_dir + os.sep + dataset_name + "_TRAIN_N.pt")
#                 torch.save(test_dataset, dataset_dir + os.sep + dataset_name + "_TEST_N.pt")
#                 torch.save(val_dataset, dataset_dir + os.sep + dataset_name + "_VAL_N.pt")
#             else:
#                 torch.save(train_dataset, dataset_dir + os.sep + dataset_name + "_TRAIN.pt")
#                 torch.save(test_dataset, dataset_dir + os.sep + dataset_name + "_TEST.pt")
#                 torch.save(val_dataset, dataset_dir + os.sep + dataset_name + "_VAL.pt")
                
#             if print_dataset_info:
#                 print("Saved loaded data to .pt file")

#     train_dataset.to(device)
#     test_dataset.to(device)
#     val_dataset.to(device)
    
#     if print_dataset_info:
#         print("Loaded data dim:")
#         print("Train:", train_dataset.data_x.shape)
#         print("Test:", test_dataset.data_x.shape)
#         print("Val:", val_dataset.data_x.shape)
#         print("Num class:", len(torch.unique(test_dataset.data_y)))
#         print("Batch size:", batch_size)
#         print("="*5)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     if return_encoder:
#         return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset, data_preprocess.label_encoder
#     else:
#         return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset

    
# def load_MIMIC_data(
#     dataset_dir,
#     normalize_data=False,
#     batch_size=256,
#     cross_val=False,
#     print_dataset_info=False,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
#     print("Loading MIMIC-III In-Hospital Mortality")
    
#     train_x = np.load(f"{dataset_dir}/train_X.npy")
#     train_y = np.load(f"{dataset_dir}/train_y.npy")
#     test_x = np.load(f"{dataset_dir}/test_X.npy")
#     test_y = np.load(f"{dataset_dir}/test_y.npy")
#     val_x = np.load(f"{dataset_dir}/val_X.npy")
#     val_y = np.load(f"{dataset_dir}/val_y.npy")
    
#     if cross_val:
#         train_x = np.concatenate([train_x, test_x, val_x], axis=0)
#         train_y = np.concatenate([train_y, test_y, val_y], axis=0)
#         train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
#         test_x, test_y = val_x.copy(), val_y.copy()
#     else:
#         val_x = np.load(f"{dataset_dir}/val_X.npy")
#         val_y = np.load(f"{dataset_dir}/val_y.npy")
#         test_x = np.load(f"{dataset_dir}/test_X.npy")
#         test_y = np.load(f"{dataset_dir}/test_y.npy")
#     print(np.unique(train_y.astype(int)), np.unique(test_y.astype(int)), np.unique(val_y.astype(int)))
#     print(np.bincount(train_y.astype(int)), np.bincount(test_y.astype(int)), np.bincount(val_y.astype(int)))
    
#     data_preprocess = DataPreprocess(train_x, train_y, test_x, test_y, val_x, val_y)
#     train_x, train_y, test_x, test_y, val_x, val_y = data_preprocess.transform(normalize=normalize_data)
    
#     train_x_tensor = torch.tensor(train_x, device=device).float()
#     train_y_tensor = torch.tensor(train_y, device=device).long()
#     test_x_tensor = torch.tensor(test_x, device=device).float()
#     test_y_tensor = torch.tensor(test_y, device=device).long()
#     val_x_tensor = torch.tensor(val_x, device=device).float()
#     val_y_tensor = torch.tensor(val_y, device=device).long()

#     train_dataset = TscDataset(train_x_tensor, train_y_tensor)
#     test_dataset = TscDataset(test_x_tensor, test_y_tensor)
#     val_dataset = TscDataset(val_x_tensor, val_y_tensor)
    
#     train_dataset.to(device)
#     test_dataset.to(device)
#     val_dataset.to(device)
    
#     if print_dataset_info:
#         print("Loaded data dim:")
#         print("Train:", train_dataset.data_x.shape)
#         print("Test:", test_dataset.data_x.shape)
#         print("Val:", val_dataset.data_x.shape)
#         print("Num class:", len(torch.unique(test_dataset.data_y)))
#         print("Batch size:", batch_size)
#         print("="*5)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
#     return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset, data_preprocess.label_encoder