import os
import gc
import math
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import dask.dataframe as dd
from Scripts.vocab import Vocabulary
from Scripts.utils import divide_chunks
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class TransactionData(Dataset):
    def __init__(self, data_list, data_dir, seq_len, flatten, return_labels, load_all=False) -> None:
        super().__init__()
        self.data_dict = {}
        self.label_dict = {}
        self.load_all = load_all
        self.data_dir = data_dir
        self.flatten = flatten
        self.return_labels = return_labels
        self.seq_len = seq_len
        self.data = data_list
        self.current_id = -1
        self.current_user_data = None
        self.current_user_label = None
        if self.load_all:
            cur = -1
            data = None
            if self.return_labels:
                label = None
            for u_id, w_id in self.data:
                if u_id != cur:
                    cur = u_id
                    data = joblib.load(os.path.join(self.data_dir, f'PreProcessed/User_Transactions/{cur}.pkl'))
                    if self.return_labels:
                        label = joblib.load(os.path.join(self.data_dir, f'PreProcessed/User_Labels/{cur}.pkl'))
                if u_id not in self.data_dict.keys():
                    self.data_dict[u_id] = {}
                self.data_dict[u_id][w_id] = data[w_id - 1]
                if self.return_labels:
                    if u_id not in self.label_dict.keys():
                        self.label_dict[u_id] = {}
                    self.label_dict[u_id][w_id] = label[w_id - 1]


    def __getitem__(self, index):
        user_id, window_id = self.data[index]
        if not self.load_all and user_id != self.current_id:
            self.current_id = user_id
            self.current_user_data = joblib.load(os.path.join(self.data_dir, f'PreProcessed/User_Transactions/{self.current_id}.pkl'))
            if self.return_labels:
                self.current_user_label = joblib.load(os.path.join(self.data_dir, f'PreProcessed/User_Labels/{self.current_id}.pkl'))
        
        if self.load_all:
            return_data = self.data_dict[user_id][window_id]
        else:
            return_data = self.current_user_data[window_id-1]
        if self.flatten:
            return_data = torch.tensor(return_data, dtype=torch.long)
        else:
            try:
                return_data = torch.tensor(return_data, dtype=torch.long).reshape(self.seq_len, -1)
            except:
                print(window_id, user_id)
        if self.return_labels:
            if self.load_all:
                return_data = (return_data, torch.tensor(self.label_dict[user_id][window_id], dtype=torch.long))
            else:
                return_data = (return_data, torch.tensor(self.current_user_label[window_id-1], dtype=torch.long))

        return return_data

    def __len__(self):
        return len(self.data)

class Data():
    def __init__(self, data_dir='./Data', model_dir='./Models', seq_len=10, stride=5, 
                nbins=10, adap_threshold=10**8, return_labels=False, 
                skip_user=False, flatten=False, ids=None):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.seq_len = seq_len
        self.stride = stride
        self.nbins = nbins
        self.return_labels = return_labels
        self.skip_user = skip_user
        self.flatten = flatten
        self.data_prep_models = {}
        self.data = []
        self.labels = []
        self.window_label = []
        self.vocab = Vocabulary(adap_thres=adap_threshold)
        self.encode_data()
        self.init_vocab()
        self.prepare_samples(ids)
        self.save_vocab()

    def encode_data(self):
        if os.path.exists(os.path.join(self.data_dir, f'PreProcessed/data.pkl')):
            return
        data_prep_path = os.path.join(self.model_dir, 'data_prep.pkl')
        num8_cols = ['Card', 'Timestamp', 'Amount', 'Use Chip', 'Merchant State', 'MCC', 'Errors?', 'Is Fraud?']
        num16_cols = ['User', 'Merchant City']
        num32_cols = ['Merchant Name', 'Zip']

        dtypes = {}
        for c in num8_cols:
          dtypes[c] = 'int8'

        for c in num16_cols:
          dtypes[c] = 'int16'

        for c in num32_cols:
          dtypes[c] = 'int32'

        if os.path.exists(data_prep_path):
            self.trans_data = pd.read_csv(os.path.join(self.data_dir, 'PreProcessed/transactions.csv'), dtype = dtypes)
            self.data_prep_models = joblib.load(data_prep_path)
            return
        
        df = pd.read_csv(os.path.join(self.data_dir, 'transactions.csv'))

        df['Zip'] = df['Zip'].fillna(0).astype(int)
        num8_col = ['Card', 'Month', 'Day']
        num16_col = ['User', 'Year']
        cat_col = ['Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 
                    'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?']
        df[num8_col] = df[num8_col].astype('int8')
        df[num16_col] = df[num16_col].astype('int16')
        df[cat_col] = df[cat_col].astype('category')
        df['Amount'] = df['Amount'].apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        df['Errors?'] = df['Errors?'].cat.add_categories('None').fillna('None')
        df['Is Fraud?'] = df['Is Fraud?'].cat.rename_categories([0, 1]).astype('int8')
        df['Merchant State'] = df['Merchant State'].cat.add_categories('None').fillna('None')
        df['Use Chip'] = df['Use Chip'].cat.add_categories('None').fillna('None')

        sub_columns = ['Errors?', 'MCC', 'Zip', 'Merchant State', 'Merchant City', 'Merchant Name', 'Use Chip']
        for col_name in tqdm(sub_columns, desc='Label Encoding'):
            col_data = df[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.data_prep_models[col_name] = col_fit
            df[col_name] = col_data

        timestamp = self.timeEncoder(df[['Year', 'Month', 'Day', 'Time']])
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.data_prep_models['Timestamp'] = timestamp_fit
        df['Timestamp'] = timestamp

        coldata = np.array(df['Timestamp'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        df['Timestamp'] = self._quantize(coldata, bin_edges)
        self.data_prep_models["Timestamp-Quant"] = [bin_edges, bin_centers, bin_widths]

        coldata = np.array(df['Amount'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        df['Amount'] = self._quantize(coldata, bin_edges)
        self.data_prep_models["Amount-Quant"] = [bin_edges, bin_centers, bin_widths]

        columns_to_select = ['User', 'Card', 'Timestamp', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?']

        joblib.dump(self.data_prep_models, data_prep_path)
        df[columns_to_select].to_csv(os.path.join(self.data_dir, 'PreProcessed/transactions.csv'), index=False)
        del df
        gc.collect()
        self.trans_data = pd.read_csv(os.path.join(self.data_dir, 'PreProcessed/transactions.csv'), dtype=dtypes)

    def init_vocab(self):
        vocab_path = os.path.join(self.model_dir, 'vocab.pkl')
        
        if os.path.exists(vocab_path):
            self.vocab = joblib.load(vocab_path)
            return
        
        column_names = list(self.trans_data.columns)
        if self.skip_user:
            column_names.remove("User")

        self.vocab.set_field_keys(column_names)

        for column in tqdm(column_names, desc='Creating Vocab'):
            unique_values = self.trans_data[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            
            if vocab_size > self.vocab.adap_thres:
                self.vocab.adap_sm_cols.add(column)
        joblib.dump(self.vocab, vocab_path)

    def save_vocab(self):
        file_name = os.path.join(self.model_dir, 'vocab.nb')
        self.vocab.save_vocab(file_name)
    
    def select_ids(self, ids):
        new_data = []
        for user_id, window_id in self.data:
            if user_id in ids:
                new_data.append([user_id, window_id])
        return new_data
    
    def prepare_samples(self, ids=None):
        trans_path = os.path.join(self.data_dir, 'trans_data.pkl')
        data_path = os.path.join(self.data_dir, f'PreProcessed/data.pkl')
        
        if os.path.exists(data_path):
            self.data = joblib.load(data_path)
            if ids:
                self.data = self.select_ids(ids)
            self.ncols = len(self.vocab.field_keys) - 2 + 1
            return
        if os.path.exists(trans_path):
            trans_data, trans_labels, columns_names = joblib.load(trans_path)
        else:
            trans_data, trans_labels, columns_names = self.user_level_data()
            joblib.dump([trans_data, trans_labels, columns_names], trans_path)
        
        if not os.path.exists(os.path.join(self.data_dir, f'PreProcessed/User_Transactions')):
            os.mkdir(os.path.join(self.data_dir, f'PreProcessed/User_Transactions'))
        if not os.path.exists(os.path.join(self.data_dir, f'PreProcessed/User_Labels')):
            os.mkdir(os.path.join(self.data_dir, f'PreProcessed/User_Labels'))

        user_idx = 0
        with tqdm(total=len(trans_data)) as pbar:
            while len(trans_data) > 0:
                global_id = 0
                user_dict = {}
                label_dict = {}
                user_row = trans_data.pop(0)
                user_row_ids = self.format_trans(user_row, columns_names)
                user_labels = trans_labels.pop(0)
                for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.stride):
                    ids = user_row_ids[jdx:(jdx + self.seq_len)]
                    ids = [idx for ids_lst in ids for idx in ids_lst]
                    user_dict[global_id] = ids
                    ids = user_labels[jdx:(jdx + self.seq_len)]
                    label_dict[global_id] = ids
                    self.data.append([user_idx, global_id])
                    global_id += 1
                joblib.dump(user_dict, os.path.join(self.data_dir, f'PreProcessed/User_Transactions/{user_idx}.pkl'))
                joblib.dump(label_dict, os.path.join(self.data_dir, f'PreProcessed/User_Labels/{user_idx}.pkl'))
                user_idx += 1
                pbar.update()
        del trans_data
        del trans_labels
        del columns_names
        gc.collect()
        self.ncols = len(self.vocab.field_keys) - 2 + 1
        joblib.dump(self.data, data_path)
            

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def user_level_data(self):
        trans_data, trans_labels = [], []
        unique_users = self.trans_data["User"].unique()
        columns_names = list(self.trans_data.columns)

        for user in tqdm(unique_users, desc='User Transactions'):
            user_data = self.trans_data[self.trans_data['User'] == user]

            user_trans, user_labels = [], []
            for _, row in user_data.iterrows():
                row = list(row)
                skip_idx = 1 if self.skip_user else 0
                user_trans.extend(row[skip_idx:-1])
                user_labels.append(row[-1])

            trans_data.append(user_trans)
            trans_labels.append(user_labels)

        if self.skip_user:
            columns_names.remove("User")

        return trans_data, trans_labels, columns_names

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        X_hm = X['Time'].str.split(':', expand=True)
        d = pd.to_numeric(pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1]))).astype(int)
        return d

    @staticmethod
    def amountEncoder(X):
        amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return amt

    @staticmethod
    def fraudEncoder(X):
        fraud = (X == 'Yes').astype(int)
        return pd.DataFrame(fraud)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.nbins, 1 / self.nbins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.nbins) - 1  # Clip edges
        return quant_inputs

