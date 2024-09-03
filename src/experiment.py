
import time

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from src.utils.splits import StratifiedTrainsPDiff
from src.utils.stability_measure import StabilityMeasure
from src.utils.progress_bar import printProgressBar


def experiment(classifiers:list, datasets:list, ps:list, max_cv_steps=30, random_states = [42, 16, 101], verbose=False, buffer=False):

    results_df = pd.DataFrame(columns=[
        "classifier",
        "dataset",
        "cv-procedure",
        "p",
        "Lustgarten",
        "Nogueira",
        "Jaccard",
        "#features_mean",
        "#features_min",
        "#features_max",
        "acc_train",
        "acc_test",
        "execution_time"
    ])

    for classifier in classifiers:
        print(f"-> Classifier: {classifier['name']} ({classifier['estimator']})")

        selector = SelectFromModel(
            estimator = classifier["estimator"],
            threshold = classifier["threshold"],
            importance_getter = classifier["importance_getter"]
        )

        for dataset in datasets:
            print(f"---> Dataset: {dataset['name']} {dataset['data']['X'].shape}")
            n_features = dataset["data"]["X"].shape[1]

            for p in ps:
                n_steps = len(random_states) * (dataset['train_size'] // p + 1)
                if (max_cv_steps is not None) and (n_steps > len(random_states)*max_cv_steps):
                    n_steps = len(random_states)*max_cv_steps
                message_cv = f"-----> trains-p-diff({dataset['train_size']}, {p})"
                message_cv += " " * (30-len(message_cv))
                step = 0
                printProgressBar(step, n_steps, prefix=message_cv, suffix='Complete', length=50)

                lustgarten_values = []
                nogueira_values = []
                jaccard_values = []
                selected_features_counts = []
                train_accs = []
                test_accs = []
                execution_times = []

                for random_state in random_states:
                    stpd = StratifiedTrainsPDiff(dataset["train_size"], p, random_state)
                    selected_features_sets = []
                    cv_step = 0
                    for train_index, test_index in stpd.split(dataset["data"]["X"], dataset["data"]["y"]):
                        cv_step += 1
                        if (max_cv_steps is not None) and (cv_step > max_cv_steps):
                            break
                        X_train = dataset["data"]["X"].iloc[train_index, :]
                        y_train = dataset["data"]["y"].iloc[train_index]
                        X_test = dataset["data"]["X"].iloc[test_index, :]
                        y_test = dataset["data"]["y"].iloc[test_index]
                        # select features
                        start_time = time.time()
                        selector.fit(X_train, y_train)
                        stop_time = time.time()
                        selected_features_indexes = selector.get_support(indices=True)
                        selected_features_sets.append(set(selected_features_indexes))
                        execution_times.append(stop_time-start_time)
                        # acc
                        cls = selector.estimator_
                        sfi = list(selected_features_indexes)
                        cls.fit(X_train.iloc[:,sfi], y_train)
                        train_accs.append(accuracy_score(y_train, cls.predict(X_train.iloc[:,sfi])))
                        test_accs.append(accuracy_score(y_test, cls.predict(X_test.iloc[:,sfi])))

                        step += 1
                        printProgressBar(step, n_steps, prefix=message_cv, suffix='Complete', length=50)

                    selected_features_counts += [len(fs) for fs in selected_features_sets]
                    lustgarten_values.append(StabilityMeasure.Lustgarten(selected_features_sets, n_features))
                    nogueira_values.append(StabilityMeasure.Nogueira(selected_features_sets, n_features))
                    jaccard_values.append(StabilityMeasure.JaccardIndex(selected_features_sets, n_features))

                new_row = pd.DataFrame({
                    "classifier": f"{classifier['name']} ({classifier['estimator']})",
                    "dataset": dataset["name"],
                    "cv-procedure": f"trains-p-diff({dataset['train_size']},{p})",
                    "p": p,
                    "Lustgarten": round(sum(lustgarten_values)/len(lustgarten_values), 4),
                    "Nogueira": round(sum(nogueira_values)/len(nogueira_values), 4),
                    "Jaccard": round(sum(jaccard_values)/len(jaccard_values), 4),
                    "#features_mean": round(sum(selected_features_counts)/len(selected_features_counts), 4),
                    "#features_min": min(selected_features_counts),
                    "#features_max": max(selected_features_counts),
                    "acc_train": round(sum(train_accs)/len(train_accs), 4),
                    "acc_test": round(sum(test_accs)/len(test_accs), 4),
                    "execution_time": round(sum(execution_times)/len(execution_times), 4)
                }, index=[0])

                if verbose:
                    print(new_row.values[0,4:])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                if buffer:
                    results_df.to_csv("results_buffer.csv", index=False)

    return results_df
