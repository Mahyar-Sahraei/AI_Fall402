import pandas as pd
import model
from sklearn.metrics import f1_score


df = pd.read_csv("NLP/nlp_test.csv")
predictions = model.predict(df)
print("F1 score of the predictions:", f1_score(df["Category"], predictions, pos_label="Politics"))