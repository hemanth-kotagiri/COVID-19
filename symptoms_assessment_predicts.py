# patient_df = pd.read_csv("./datasets/dataset_1/patient.csv")

# patient_df = patient_df[["sex", "birth_year", "country", "confirmed_date"]]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_path = '.\datasets\symptoms data\covid19-symptoms-dataset.csv'

total_data = pd.read_csv(data_path,dtype='unicode')

total_features = list(total_data.columns)
# print(total_data.head())
# print(total_features)

wanted_features = ['Dry Cough', 'High Fever', 'Sore Throat', 'Difficulty in breathing']

#print(wanted_features)

cleaned_data = total_data[wanted_features]
cleaned_data = cleaned_data
y = total_data['Infected with Covid19']
#print(cleaned_data.head())

def symptoms_check():
    curr = []
    print("Please give symptoms on a scale of 1-20")
    print("0 for No symptom")
    print("1-5 : Low")
    print("6-12: Moderate")
    print("13-20: High")

    for feature in wanted_features:
        response = int(input("{} : ".format(feature)))
        if 0 <= response <= 20:
            curr.append(response)
        else:
            print("that is not a valid input.")
            exit(0)
    curr = np.array(curr)
    df = pd.DataFrame(curr.reshape(-1, len(curr)),columns=wanted_features)
    return df

# X_train, X_val, y_train, y_val = train_test_split(cleaned_data, y, random_state=0)

model = RandomForestClassifier()
model.fit(cleaned_data, y)

asked_sym = symptoms_check()

predictions = model.predict(asked_sym)

if predictions == '0':
    print("\nThe model potentially predicts that you do not have COVID-19\n")
else:
    print("\nThe model predicts that you may have COVID-19. Please do immediately reach out with medical professionals.\n")
