# import pandas and model_selection module for scikit-learn
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Training data is in a csv file called train.csv
    df = pd.read_csv("train.csv")

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows in the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df.targets.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new Kfold column
    for fold, (train_idx, val_idx) in enumerate(df.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)