from ES_prediction import *

import pandas as pd

enzymes = pd.read_csv("esp_unique_enzymes.csv")

print("Step 2/3: Calculating numerical representations for all enzymes.")
results = calculate_esm1b_ts_vectors(enzymes)

with open("proteins_esp.pkl", 'wb') as file:
    pickle.dump(results, file)