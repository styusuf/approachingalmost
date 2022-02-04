# stratified kfold cross validation for regression tasks

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # randomize the rows in the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule. I take the floor of the value, you can also just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column. Instead of targets, we will split by bins!

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[val_idx, 'kfold'] = fold

    # drop the bins column
    data = data.drop("bins", axis=1)

    return data

if __name__ == "__main__":
    # we create a sample dataset with 15000 samples and 100 features/ 1 target
    X, y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)

    # create a dataframe out of our numpy arrays
    df = pd.DataFrame(X, columns=["f_{i}" for i in range(X.shape[1])])
    df.loc[:, "target"] = y

    # create folds
    df = create_folds(df)