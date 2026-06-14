import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/heart_disease_cleveland.csv")

    print("Dataset shape (rows, columns):", df.shape)
    print("\nMissing values per column (before cleaning):")
    print(df.isnull().sum())
    df = df.dropna()

    print("\nDataset shape after dropping missing rows:", df.shape)

    print("\nTarget value counts (0 = no disease, 1 = disease):")
    print(df["target"].value_counts())

    return df

def prepare_data(df):
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("\nTraining set shape:", X_train_scaled.shape)
    print("Testing set shape:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)