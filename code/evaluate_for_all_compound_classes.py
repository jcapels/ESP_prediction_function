


import os
import pandas as pd

from optimize_hyperparameters import evaluate_model, prepare_dataset, train_model


def train_and_evaluate(param, splits, dataset, compounds_split_datasets, sim=20, name="results_esp.csv", proteins=True):
    from copy import copy

    for seed in range(5):
        # Assuming splits is a list of tuples, each containing (train_ids, val_ids, test_ids)
        for train_ids, val_ids, test_ids in splits:

            # convert sets into pandas Series
            if not isinstance(train_ids, pd.Series):
                train_ids = pd.Series(list(train_ids))
            if not isinstance(val_ids, pd.Series):
                val_ids = pd.Series(list(val_ids))

            train_ids = pd.concat((train_ids, val_ids))

            # Prepare training data
            X_train, y_train = prepare_dataset(dataset, train_ids, random_seed=seed, proteins=proteins)

            # Train the model
            params = copy(param)
            model = train_model(X_train, y_train, seed=seed, param=params)

            # Prepare validation data
            # Prepare test data

            for class_ in compounds_split_datasets:

                test_ids_ = [id_ for id_ in compounds_split_datasets[class_] if id_ in test_ids]
                if len(test_ids_) == 0:
                    continue
                X_test, y_test = prepare_dataset(dataset, test_ids_, proteins=proteins)
                try:
                    # Evaluate the model on test data
                    test_metrics = evaluate_model(model, X_test, y_test)

                    test_metrics["seed"] = seed
                    test_metrics["similarity"] = sim

                    df = pd.DataFrame([test_metrics])
                    if os.path.exists(f"{name}_{class_}.csv"):
                        results = pd.read_csv(f"{name}_{class_}.csv")
                    else:
                        results = pd.DataFrame()
                    results = pd.concat([results, df], ignore_index=True)
                    results.to_csv(f"{name}_{class_}.csv", index=False)
                except ValueError:
                    pass

def evaluate_for_compound_classes(pickle_format, sim):

    import pickle
    with open(f"results_stereo/best_params_0_{pickle_format}_compounds_train_val_test.pkl", "rb") as f:
        best = pickle.load(f)

    dataset = pd.read_pickle("dataset.pkl")

    with open(f"splits/splits_compounds_0{pickle_format}_v4.pkl", "rb") as f:
        splits = pickle.load(f)

    with open(f"splits/pathway_to_compounds_split_0{pickle_format}_v4.pkl", "rb") as f:
        class_splits = pickle.load(f)

    train_and_evaluate(best, splits, dataset, 
                       compounds_split_datasets=class_splits,name=f"results_esp_compounds_test", 
                       sim = sim, 
                       proteins=False)

if __name__ == "__main__":

    evaluate_for_compound_classes("2", 20)
    evaluate_for_compound_classes("4", 40)
    evaluate_for_compound_classes("6", 60)
    evaluate_for_compound_classes("8", 80)
    



