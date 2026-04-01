from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import re

def op_factory():
    REDUCE_OPS = {
        "add": lambda x: x.sum(axis=1),
        "mul": lambda x: np.prod(x, axis=1),
        "mean": lambda x: x.mean(axis=1),
        "max": lambda x: x.max(axis=1),
        "min": lambda x: x.min(axis=1),
    }

    INTERACT_OPS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / (b + 1e-6),
    }

    return REDUCE_OPS, INTERACT_OPS

REDUCE_OPS, INTERACT_OPS = op_factory()

def global_sanitize(X):
    if hasattr(X, "columns"):
        X.columns = [re.sub(r"__+", "_", re.sub(r"\W+", "_", col)).strip("_") for col in X.columns]
    return X

class InteractionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, interactions):
        self.interactions = interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for cols, reduce_op, interact_col, interact_op, feat_name in self.interactions:
            if not all(c in X.columns for c in cols):
                continue
            
            base = X[cols]
            feat = REDUCE_OPS[reduce_op](base.values) if reduce_op else base.iloc[:, 0].to_numpy()

            if interact_col and interact_col in X.columns:
                feat = INTERACT_OPS[interact_op](feat, X[interact_col].to_numpy().ravel())

            name = global_sanitize(feat_name) if feat_name else global_sanitize(f"{reduce_op or ''}_{'_'.join(cols)}")
            X[name] = feat
        return X

    def get_feature_names_out(self, input_features=None):
        out = list(input_features)
        for cols, reduce_op, interact_col, interact_op, feat_name in self.interactions:
            if all(c in input_features for c in cols):
                name = global_sanitize(feat_name) if feat_name else global_sanitize(f"{reduce_op or ''}_{'_'.join(cols)}")
                out.append(name)
        return np.array(out)
    
class Theme_Gap(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.history_ = {} 
        self.final_idx_ = 0

    def _get_timeline(self, year, session):
        mapping = {"Jan/Feb": 0, "May/June": 1, "Oct/Nov": 2}
        return (year * 3) + mapping.get(session, 0)

    def fit(self, X, y=None):
        if y is None: return self
        
        temp = X.copy()
        temp["App_Next"] = y
        temp["TimeID"] = temp.apply(lambda x: self._get_timeline(x["Cutoff_Year"], x["Cutoff_Session"]), axis=1)
        
        self.history_ = temp[temp["App_Next"] == 1].groupby("Primary_Theme")["TimeID"].max().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X["TimeID"] = X.apply(lambda x: self._get_timeline(x["Cutoff_Year"], x["Cutoff_Session"]), axis=1)
        X["Theme_Gap"] = 99

        def calculate_gap(row):
            theme = row["Primary_Theme"]
            current_time = row["TimeID"]
            if theme in self.history_:
                last_time = self.history_[theme]
                return current_time - last_time
            return 99
        
        X["Theme_Gap"] = X.apply(calculate_gap, axis=1)

        return X.drop(columns=["TimeID"])
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.append(input_features, "Theme_Gap")
    
class StreamThemeEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, m=3):
        self.m = m
        self.st_mappings_ = {}
        self.st_counts = {}
        self.tt_mappings_ = {}
        self.global_mean_ = 0
        self.totaL_sessions_ = 0

    def fit(self, X, y):
        X = X.copy()
        X['target'] = y
        self.global_mean_ = y.mean()
        self.total_sessions_ = len(X["Cutoff_Session"].unique())
        
        st_key_series = X['Paper_Stream'] + "_" + X['Primary_Theme']
        tt_key_series = X["Text_Type"]

        st_stats = X.assign(st_key=st_key_series).groupby('st_key')['target'].agg(['sum', 'count'])
        for st_key, st_row in st_stats.iterrows():
            self.st_mappings_[st_key] = (st_row['sum'] + self.global_mean_ * self.m) / (st_row['count'] + self.m)
            self.st_counts[st_key] = st_row['count']
        
        tt_stats = X.assign(tt_key=tt_key_series).groupby('tt_key')['target'].agg(['sum', 'count'])
        for tt_key, tt_row in tt_stats.iterrows():
            self.tt_mappings_[tt_key] = (tt_row['sum'] + self.global_mean_ * self.m) / (tt_row['count'] + self.m)

        return self

    def transform(self, X):
        X = X.copy()

        st_key = X['Paper_Stream'] + "_" + X['Primary_Theme']
        st_win = st_key.map(self.st_mappings_).fillna(self.global_mean_)
        tt_win = X['Text_Type'].map(self.tt_mappings_).fillna(self.global_mean_)
        
        X["Theme_Likelihood"] = (X['Theme_Gap'] * st_win) * (X["Theme_Gap"]> 1)
        X["Text_Strength"] = tt_win - self.global_mean_
        #X["Critical_Win_Rate"] = (X["Sessions_Since_Last"] >= 5).astype(int) * st_win
        X['Is_Overdue'] = (X['Sessions_Since_Last'] >= 5).astype(int)
        
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        new_cols = ["Theme_Likelihood", "Text_Strength", "Is_Overdue"]
        return np.append(input_features, new_cols)

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.feature_names_in_ = []
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, "columns") else []
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                X[col] = np.log1p(X[col])
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array(self.feature_names_in_ if self.feature_names_in_ is not None else [], dtype=object)

class GlobalSessionsSinceLast(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Global_Sessions_Since_Last"] = X.groupby(["Text_Title", "Cutoff_Year", "Cutoff_Session"])["Sessions_Since_Last"].transform("min")
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.append(input_features, ["Global_Sessions_Since_Last"])