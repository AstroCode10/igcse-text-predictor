import pandas as pd
import numpy as np

# REDUCE and INTERACT ops from the notebook
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

class InteractionAdder:
    """Small, robust version of the user's transformer for testing.

    Expects interaction tuples of the form:
      (cols, reduce_op, interact_col, interact_op, feat_name)

    Behavior:
      - If reduce_op is None and cols contains a single column, use that
        column's values directly (no reduction).
      - If reduce_op is not None, use REDUCE_OPS[reduce_op].
      - If interact_col is not None, check that interact_op is a valid key
        in INTERACT_OPS and apply it.
    """

    def __init__(self, interactions):
        self.interactions = interactions
        self.feature_names_in_ = []
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns)
        return self

    def make_name(self, cols, reduce_op, interact_col, interact_op):
        created_name = f"{reduce_op}(" + ",".join(cols) + ")" if reduce_op is not None else cols[0]
        if interact_col is not None:
            return f"{created_name}_{interact_op}_{interact_col}"
        return created_name

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def transform(self, X):
        X = X.copy()

        for cols, reduce_op, interact_col, interact_op, feat_name in self.interactions:
            base = X[cols]

            # handle reduce_op == None (use single column as-is)
            if reduce_op is None:
                if base.shape[1] != 1:
                    raise ValueError("reduce_op is None but multiple cols were provided")
                feat = base.iloc[:, 0].to_numpy()
            else:
                if reduce_op not in REDUCE_OPS:
                    raise KeyError(f"Unknown reduce_op: {reduce_op}")
                feat = REDUCE_OPS[reduce_op](base.values)

            # handle interaction
            if interact_col is not None:
                if interact_op not in INTERACT_OPS:
                    raise KeyError(f"Unknown interact_op: {interact_op}")
                if interact_col not in X.columns:
                    raise KeyError(f"Interaction column '{interact_col}' not found in X")
                feat = INTERACT_OPS[interact_op](feat, X[interact_col].to_numpy())

            # determine name
            if feat_name is not None:
                name = feat_name
            else:
                name = self.make_name(cols, reduce_op, interact_col, interact_op)

            X[name] = feat
            self.feature_names_out_.append(name)

        return X


if __name__ == '__main__':
    # small dataframe with columns used in the notebook
    df = pd.DataFrame({
        'Appd_Last_1': [1, 0, 2],
        'Appd_Last_2': [0, 1, 1],
        'Appd_Last_3': [1, 1, 0],
        'Num_Stream_App': [5, 2, 0],
        'App_Total': [10, 5, 2],
        'Sessions_Since_Last': [3, 4, 1],
    })

    # the original (buggy) interactions from the notebook
    buggy_interactions = [
        (['Appd_Last_1', 'Appd_Last_2'], 'add', 'Appd_Last_3', 'add', 'Recent_Frequency'),
        (['Num_Stream_App'], None, 'div', 'App_Total', 'Stream_App_Ratio'),
        (['Sessions_Since_Last', 'App_Total'], 'mul', None, None, None),
    ]

    print('\nTrying to transform with the buggy interactions (expected to fail)...')
    try:
        tr = InteractionAdder(buggy_interactions)
        tr.fit(df)
        out = tr.transform(df)
        print(out.head())
    except Exception as e:
        print('ERROR:', type(e).__name__, e)

    # corrected ordering for the second tuple (cols, reduce_op, interact_col, interact_op, feat_name)
    fixed_interactions = [
        (['Appd_Last_1', 'Appd_Last_2'], 'add', 'Appd_Last_3', 'add', 'Recent_Frequency'),
        (['Num_Stream_App'], None, 'App_Total', 'div', 'Stream_App_Ratio'),
        (['Sessions_Since_Last', 'App_Total'], 'mul', None, None, None),
    ]

    print('\nTrying to transform with the fixed interactions (should succeed)...')
    tr2 = InteractionAdder(fixed_interactions)
    tr2.fit(df)
    out2 = tr2.transform(df)
    print(out2[['Recent_Frequency', 'Stream_App_Ratio', 'mul(Sessions_Since_Last,App_Total)']])
    print('\nFeature names out:', tr2.get_feature_names_out())
