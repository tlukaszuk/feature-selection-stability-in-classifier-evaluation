import os
import pandas as pd


class Datasets():
   
    datasets = {
        "Breast": "Breast_GSE70947.zip",
        "Colorectal": "Colorectal_GSE44076.zip",
        "Leukemia": "Leukemia_GSE63270.zip",
        "Liver": "Liver_GSE76427.zip",
        "Prostate": "Prostate_GSE6919_U95B.zip",
        "Renal": "Renal_GSE53757.zip",
        "Throat": "Throat_GSE42743.zip",
    }

    def get(name:str) -> dict:
        dir = os.environ.get('DATA_PATH','data')
        file = Datasets.datasets[name]
        path = os.path.join(dir, file)
        df = pd.read_csv(path, sep=",")
        X = df.iloc[:,2:]
        y = df.iloc[:,1]
        y = y.apply(lambda x: 0 if x=="normal" else 1)
        return {"X": X, "y":y}


if __name__ == "__main__":
    
    for name in ["Breast", "Colorectal", "Leukemia", "Liver", "Prostate", "Renal", "Throat"]:
        ds = Datasets.get(name)
        print("---------------------")
        print(name)
        print(f"X.shape={ds['X'].shape}")
        print(f"y.shape={ds['y'].shape}")
        print(f"{ds['y'].value_counts()}")