
import pickle
import pandas as pd

import datetime
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from hurry.filesize import size

from ES_prediction import *

from optimize_hyperparameters import train_and_report_loss
from hyperopt import fmin, tpe, hp, Trials, rand
from tqdm import tqdm
from tqdm import tqdm
from enzyme_representations import calculate_esm1b_ts_vectors
from metabolite_preprocessing import metabolite_preprocessing

import os
import sys
from contextlib import contextmanager

import logging
logger = logging.getLogger('rdkit')
logger.addHandler(logging.NullHandler())
logger.propagate = False

import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def generate_features(dataset_csv, name):

    dataset = pd.read_csv(dataset_csv)
    substrates = dataset.drop_duplicates(subset=["Substrate ID"]).loc[:, "SMILES"]
    print(substrates.shape)
    enzymes = dataset.drop_duplicates(subset=["Enzyme ID"])

    print("Step 1/3: Calculating numerical representations for all metabolites.")
    df_met = metabolite_preprocessing(metabolite_list = substrates)

    print("Step 2/3: Calculating numerical representations for all enzymes.")
    results = calculate_esm1b_ts_vectors(enzymes)

    with open(f"proteins_esp_{name}.pkl", 'wb') as file:
        pickle.dump(results, file)

    with open(f"compounds_esp_{name}.pkl", 'wb') as file:
        pickle.dump(df_met, file)

def benchmark_optimize_hyperparameters():
    #Defining search space for hyperparameter optimization
    space_gradient_boosting = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
        "max_depth": hp.choice("max_depcth", [9,10,11,12,13]),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "max_delta_step": hp.uniform("max_delta_step", 0, 5),
        "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
        "num_rounds":  hp.uniform("num_rounds", 200, 400),
        "weight" : hp.uniform("weight", 0.1,0.33)}

    trials = Trials()

    tracemalloc.start()
    start = time.time()

    for i in tqdm(range(1,2000)):
        best = fmin(fn = train_and_report_loss, space = space_gradient_boosting,
                    algo = rand.suggest, max_evals = i, trials = trials)
        
    results = pd.DataFrame()
    end = time.time()
    print("Time spent: ", end - start)
    print("Memory needed: ", tracemalloc.get_traced_memory())

    results = pd.concat((results, 
                            pd.DataFrame({
                                            "time": [str(datetime.timedelta(seconds=end - start))], 
                                            "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                            ignore_index=True, axis=0)
    tracemalloc.stop()
    results.to_csv("benchmark_optimize_hyperparameters.csv")

def benchmark_resources_generating_features():
    pipelines = [generate_features]
    datasets = ["efficiency_performance/dataset_100_100.csv", "efficiency_performance/dataset_300_1000.csv", "efficiency_performance/dataset_700_5000.csv", 
                "efficiency_performance/dataset_7000.csv", "curated_dataset.csv"]
    
    
    if os.path.exists("benchmark_results.csv"):
        results = pd.read_csv("benchmark_results.csv")
    else:
        results = pd.DataFrame()

    i=0
    for pipeline in pipelines:
        for dataset in datasets:
            dataset_df = pd.read_csv(dataset)
            tracemalloc.start()
            start = time.time()
            generate_features(dataset, str(i))
            i+=1

            end = time.time()
            print("Time spent: ", end - start)
            print("Memory needed: ", tracemalloc.get_traced_memory())
            unique_substrates_dataset = np.unique(dataset_df["Substrate ID"])
            num_unique_substrates = len(unique_substrates_dataset)
            unique_enzymes_dataset = np.unique(dataset_df["Enzyme ID"])
            num_unique_enzymes = len(unique_enzymes_dataset)
            num_rows = dataset_df.shape[0]

            results = pd.concat((results, 
                                    pd.DataFrame({"pipeline": [pipeline.__name__], 
                                                  "unique_enzymes": [num_unique_enzymes],
                                                  "unique_substrates": [num_unique_substrates],
                                                  "num_pairs": [num_rows],
                                                  "time": [str(datetime.timedelta(seconds=end - start))], 
                                                  "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                  ignore_index=True, axis=0)
            tracemalloc.stop()

            results.to_csv("benchmark_results.csv", index=False)

def inference(dataset_path):

    import pandas as pd

    df = pd.read_csv(dataset_path)

    ESP_predicton(df["SMILES"].tolist(), df["Sequence"].tolist())


def benchmark_resources_inference():
    datasets = ["efficiency_performance/dataset_100_100.csv", "efficiency_performance/dataset_300_1000.csv", "efficiency_performance/dataset_700_5000.csv", 
                "efficiency_performance/dataset_7000.csv", "curated_dataset.csv"]
    
    
    if os.path.exists("benchmark_results_inference.csv"):
        results = pd.read_csv("benchmark_results_inference.csv")
    else:
        results = pd.DataFrame()

    for dataset in datasets:
        dataset_df = pd.read_csv(dataset)
        tracemalloc.start()
        start = time.time()
        inference(dataset)

        end = time.time()
        print("Time spent: ", end - start)
        print("Memory needed: ", tracemalloc.get_traced_memory())
        unique_substrates_dataset = np.unique(dataset_df["Substrate ID"])
        num_unique_substrates = len(unique_substrates_dataset)
        unique_enzymes_dataset = np.unique(dataset_df["Enzyme ID"])
        num_unique_enzymes = len(unique_enzymes_dataset)
        num_rows = dataset_df.shape[0]

        results = pd.concat((results, 
                                pd.DataFrame({ 
                                                "unique_enzymes": [num_unique_enzymes],
                                                "unique_substrates": [num_unique_substrates],
                                                "num_pairs": [num_rows],
                                                "time": [str(datetime.timedelta(seconds=end - start))], 
                                                "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                                ignore_index=True, axis=0)
        tracemalloc.stop()

        results.to_csv("benchmark_results_inference.csv", index=False)

# Usage
with suppress_stderr():
    # benchmark_resources_generating_features()
    benchmark_resources_inference()