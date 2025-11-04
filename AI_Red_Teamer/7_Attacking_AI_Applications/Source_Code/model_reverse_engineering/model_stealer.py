import requests, json, random, joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


N_SAMPLES = 100

MIN_FLIPPER_LENGTH = 150
MAX_FLIPPER_LENGTH = 250

MIN_BODY_MASS = 2500
MAX_BODY_MASS = 6500

CLASSIFIER_URL = "http://94.237.57.115:50142"

samples = {
    "Flipper Length (mm)": [],
    "Body Mass (g)": []
}

for i in range(N_SAMPLES):
    samples["Flipper Length (mm)"].append(random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH))
    samples["Body Mass (g)"].append(random.uniform(MIN_BODY_MASS, MAX_BODY_MASS))

samples_df = pd.DataFrame(samples)

print(samples_df.head())

predictions = {"species": []}

for i in range(N_SAMPLES):
    sample = {
                "flipper_length": samples["Flipper Length (mm)"][i],
                "body_mass": samples["Body Mass (g)"][i]
            }

    prediction = json.loads(requests.get(CLASSIFIER_URL, params=sample).text).get("result")
    predictions["species"].append(prediction)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.head())

# Replicating the Model: The final step in reverse engineering the 
# original model is training the surrogate model on the data we obtained in the previous step. 

surrogate_model = make_pipeline(StandardScaler(), LogisticRegression())
surrogate_model.fit(samples_df, predictions_df)

# save classifier to a file
joblib.dump(surrogate_model, 'surrogate.joblib')

# Evaluation: To submit the surrogate model to the lab, we can upload it to the lab's /model endpoint:
with open('surrogate.joblib', 'rb') as f:
    file = f.read()

r = requests.post(CLASSIFIER_URL + '/model', files={'file': ('surrogate.joblib', file)})

print(json.loads(r.text))