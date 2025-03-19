#%%
from nltk.tokenize import sent_tokenize

from lib.config.config_loader import ConfigLoader

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from tqdm import tqdm

config = ConfigLoader().load_config()
tqdm.pandas()
#%%
input_dim = config['models']['input_dim']
#%%
df = pd.read_csv('./data/processed/reports_labeled.csv')

df.reset_index(drop=True, inplace=True)
df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))

train_df = df[df['year'] <= 2019].copy()
test_df = df[df['year'] > 2019].copy()
#%%
train_corpus = [sentence for sentences in train_df['sentences'] for sentence in sentences]

vectorizer = TfidfVectorizer(max_features=input_dim, stop_words='english')
vectorizer.fit(train_corpus)

def get_tfidf_embeddings(sentence_list):
    if not type(sentence_list) == list:
        sentence_list = [sentence_list]
    embeddings = vectorizer.transform(sentence_list)
    return embeddings

print("Train MDA: ")
train_df['tfidf_mda'] = train_df['mda'].progress_apply(get_tfidf_embeddings)
print("Test MDA: ")
test_df['tfidf_mda'] = test_df['mda'].progress_apply(get_tfidf_embeddings)

#%%
X_train, y_train = train_df['tfidf_mda'], train_df['label'].to_numpy()
X_test, y_test = test_df['tfidf_mda'], test_df['label'].to_numpy()

X_train = [x.toarray()[0] for x in X_train]
X_test = [x.toarray()[0] for x in X_test]
#%%
models = {
    "Gaussian NB": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
}

predictions = {}
#%%
for name, model in models.items():
    print("Training " + name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions[name] = {
        'y_pred': y_pred,
        'y_test': y_test,
    }

#%%
def evaluate(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return accuracy, precision, recall, f1, tp, tn, fp, fn
#%%
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "TP": [],
    "TN": [],
    "FP": [],
    "FN": []
}
for name, pred_dict in predictions.items():
    print("Evaluating " + name)

    accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(pred_dict['y_test'], pred_dict['y_pred'])
    results["Model"].append(name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1"].append(f1)
    results["TP"].append(tp)
    results["TN"].append(tn)
    results["FP"].append(fp)
    results["FN"].append(fn)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1} | TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("#"*100)
results = pd.DataFrame(results)
#%%
results.to_csv('./outputs/ml_results.csv', index=False)
#%%
