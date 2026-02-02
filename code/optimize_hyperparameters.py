from ES_prediction import *


def prepare_dataset(df, ids, random_seed=None, proteins=True):
    if proteins:
        val_dataset = df[df["Enzyme ID"].isin(ids)]
        val_dataset = val_dataset.dropna(subset=["enzyme rep"])
        val_dataset = val_dataset.dropna(subset=["GNN FP"])
    else:
        val_dataset = df[df["Substrate ID"].isin(ids)]
        val_dataset = val_dataset.dropna(subset=["GNN FP"])
        val_dataset = val_dataset.dropna(subset=["enzyme rep"])

    if random_seed != None:
        val_dataset = val_dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    X = calculate_xgb_input_matrix(val_dataset)
    y = np.array(val_dataset["Binding"]) 
    return X, y


def train_model(X, y, seed, param):

    try:

        print("Training model with parameters:", param)
        
        num_round = param["num_rounds"]
        param["tree_method"] = "gpu_hist"
        param["sampling_method"] = "gradient_based"
        param['objective'] = 'binary:logistic'
        param["seed"] = seed
        weights = np.array([param["weight"] if binding == 0 else 1.0 for binding in y])
        
        del param["num_rounds"]
        del param["weight"]
        
        param['gpu_id'] = 2
        import xgboost as xgb

        feature_names =  ["GNN rep_" + str(i) for i in range(100)]
        feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
        dX = xgb.DMatrix(X, feature_names =feature_names, label = np.array(y), weight = weights)
        bst = xgb.train(param, dX, int(num_round), verbose_eval=1)
        return bst
    except Exception as e:
        print("Error during model training:", e)
        return None

import pickle
def train_and_report_loss_compounds(param):

    dataset = pd.read_pickle("dataset_no_stereo.pkl")

    with open("./splits/splits_compounds_04_v4.pkl", "rb") as f:
        splits = pickle.load(f)

    try:
        for train_ids, val_ids, test_ids in splits:

            # Prepare training data
            X_train, y_train = prepare_dataset(dataset, train_ids, random_seed=42, proteins=False)

            X_val, y_val = prepare_dataset(dataset, val_ids, random_seed=42, proteins=False)

            # Train the model
            model = train_model(X_train, y_train, 42, param)

            # Prepare validation data
            # Prepare test data
            feature_names =  ["GNN rep_" + str(i) for i in range(100)]
            feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
            dX = xgb.DMatrix(X_val, feature_names =feature_names, label = np.array(y_val))
            y_prob = np.round(model.predict(dX))

            false_positive = 100*(1-np.mean(np.array(y_val)[y_prob == 1]))
            false_negative = 100*(np.mean(np.array(y_val)[y_prob == 0]))
        
        return (2*(false_negative**2) + false_positive**1.3)
    except Exception as e:
        print("Error during training and evaluation:", e)
        return float('inf')

from sklearn.metrics import *

def evaluate_model(model, X, y, threshold=0.5):
    feature_names =  ["GNN rep_" + str(i) for i in range(100)]
    feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
    dX = xgb.DMatrix(X, feature_names =feature_names, label = np.array(y))
    y_prob = model.predict(dX)

    y_pred = (y_prob >= threshold).astype(int)  # Convert probabilities to binary predictions

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc,
        'mcc': mcc
    }

def train_and_evaluate(param, splits, dataset, name="results_esp.csv", proteins=True):
    from copy import copy

    file_exists=False
    results = pd.DataFrame()
    for seed in range(5):
        fold_idx=0
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
            X_test, y_test = prepare_dataset(dataset, test_ids, proteins=proteins)

            # Evaluate the model on test data
            test_metrics = evaluate_model(model, X_test, y_test)

            test_metrics["seed"] = seed
            test_metrics["fold"] = fold_idx
            fold_idx+=1

            df = pd.DataFrame([test_metrics])
            results = pd.concat([results, df], ignore_index=True)
            results.to_csv(name, index=False)

from hyperopt import fmin, tpe, hp, Trials, rand
from tqdm import tqdm

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

import pickle
def train_and_report_loss(param):

    dataset = pd.read_pickle("dataset_no_stereo.pkl")

    with open("./splits/splits_compounds_02_v4.pkl", "rb") as f:
        splits = pickle.load(f)

    try:
        for train_ids, val_ids, test_ids in splits:

            # Prepare training data
            X_train, y_train = prepare_dataset(dataset, train_ids, random_seed=42, proteins=False)

            X_val, y_val = prepare_dataset(dataset, val_ids, random_seed=42, proteins=False)

            # Train the model
            model = train_model(X_train, y_train, 42, param)

            # Prepare validation data
            # Prepare test data
            feature_names =  ["GNN rep_" + str(i) for i in range(100)]
            feature_names = feature_names + ["ESM1b_" + str(i) for i in range(1280)]
            dX = xgb.DMatrix(X_val, feature_names =feature_names, label = np.array(y_val))
            y_prob = np.round(model.predict(dX))

            false_positive = 100*(1-np.mean(np.array(y_val)[y_prob == 1]))
            false_negative = 100*(np.mean(np.array(y_val)[y_prob == 0]))
        
        return (2*(false_negative**2) + false_positive**1.3)
    except Exception as e:
        print("Error during training and evaluation:", e)
        return float('inf')


for i in tqdm(range(1,2000)):
    best = fmin(fn = train_and_report_loss, space = space_gradient_boosting,
                algo = rand.suggest, max_evals = i, trials = trials)
    
import pickle
with open("best_params_0_2_compounds.pkl", "wb") as f:
    pickle.dump(best, f)

import pickle
with open("best_params_0_2_compounds.pkl", "rb") as f:
    best = pickle.load(f)

dataset = pd.read_pickle("dataset_no_stereo.pkl")

with open("./splits/splits_compounds_02_v4.pkl", "rb") as f:
    splits = pickle.load(f)

train_and_evaluate(best, splits, dataset, name="results_esp_compounds_20_test.csv", proteins=False)