import os
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from Scripts.vocab import Vocabulary
from Scripts.utils import divide_chunks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Data:
    def __init__(self, data_dir='./Data', model_dir='./Models', seq_len=10, stride=5, nbins=10, adap_threshold=10**8, return_labels=False, skip_user=False, flatten=False):
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
        self.prepare_samples()
        self.save_vocab()

    def encode_data(self):
        data_prep_path = os.path.join(self.model_dir, 'data_prep.pkl')
        if os.path.exists(data_prep_path):
            self.trans_data = pd.read_csv(os.path.join(self.data_dir, 'PreProcessed/transactions.csv'))
            self.data_prep_models = joblib.load(data_prep_path)
            return
        
        df = pd.read_csv(os.path.join(self.data_dir, 'transactions.csv'))
        df['Errors?'] = self.nanNone(df['Errors?'])
        df['Is Fraud?'] = self.fraudEncoder(df['Is Fraud?'])
        df['Zip'] = self.nanZero(df['Zip'])
        df['Merchant State'] = self.nanNone(df['Merchant State'])
        df['Use Chip'] = self.nanNone(df['Use Chip'])
        df['Amount'] = self.amountEncoder(df['Amount'])

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
        self.trans_data = df[columns_to_select]

        joblib.dump(self.data_prep_models, data_prep_path)
        self.trans_data.to_csv(os.path.join(self.data_dir, 'PreProcessed/transactions.csv'), index=False)
        del df

    def init_vocab(self):
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

    def prepare_samples(self):
        trans_data, trans_labels, columns_names = self.user_level_data()
        del self.trans_data
        for user_idx in tqdm(len(trans_data), desc='Prepare Samples'):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)
            user_labels = trans_labels[user_idx]
            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]
                self.data.append(ids)
            for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                fraud = 0
                if len(np.nonzero(ids)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)
        del trans_data
        del trans_labels
        del columns_names
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)
            

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            if self.mlm:
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def user_level_data(self):
        trans_data, trans_labels = [], []
        unique_users = self.trans_data["User"].unique()
        columns_names = list(self.trans_data.columns)

        for user in tqdm(unique_users, desc='User Transactions'):
            cond = self.trans_data['User'] == user
            user_data = self.trans_data[cond]
            self.trans_data.drop(cond, axis=0, inplace=True)

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

    def save_vocab(self):
        file_name = os.path.join(self.model_dir, 'vocab.nb')
        self.vocab.save_vocab(file_name)

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
        return pd.DataFrame(d)

    @staticmethod
    def amountEncoder(X):
        amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return pd.DataFrame(amt)

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

