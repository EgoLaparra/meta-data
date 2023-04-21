import os
import sys
import json
import random
from joblib import dump, load
import argparse
import regex as re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error

from rulebased_sort import sort


def date_to_difference(df):
    df["DATE"] = pd.to_datetime(df["DATE"], format="%B %Y")
    mean_date = df["DATE"].mean().replace(day=1)
    df["DATE"] = 12 * (df["DATE"].dt.year - mean_date.year) + (df["DATE"].dt.month - mean_date.month)
    return df, mean_date


def difference_to_date(df, mean_date):
    df["DATE"] = mean_date + df["DATE"].round().apply(lambda x: pd.DateOffset(months=x))
    df["DATE"] = df["DATE"].dt.strftime('%B %Y')
    return df


def multi_decision_function(clf, input_x):
    decisions = []
    for estimator in clf.estimators_:
        decisions.append(
            estimator.decision_function(input_x)
        )
    return np.array(decisions).transpose()


def data(df, args, pred_mode=False):
    data_x = []
    data_y = []
    data_ids = []
    if not pred_mode:
        df = df[~df[args.task].isna()]
    if args.task == "DATE" and args.regression:
        df, mean_date = date_to_difference(df)
        dump(mean_date, os.path.join(args.save_model, "mean_date.joblib"))
    for row in df.iterrows():
        row = row[1].fillna("")
        fileid = row["id"]
        filename = row["filename"]
        jpath = os.path.join(args.json_dir, filename + ".json")
        assert os.path.exists(jpath)
        with open(jpath, encoding="ascii", errors="ignore") as jfile:
            jdata = json.load(jfile)
            volumes = jdata["volumes"]
            if args.alphanum:
                volumes = sorted(volumes, key=lambda x: x["name"])
            elif args.random:
                random.seed(0)
                random.shuffle(volumes)
            else:
                volumes = sort(volumes, key=lambda x: x["name"])
            if args.volumes is not None:
                num_volumes = min(args.volumes, len(volumes))
                volumes = volumes[:num_volumes]
            instring = "".join(volume["text"] for volume in volumes)
            if args.trunc is not None:
                instring = instring[:args.trunc]
            if args.pages is not None:
                doc_pages = [page for volume in jdata["volumes"] for page in volume["pages"]]
                num_page = min(args.pages, len(doc_pages)) - 1
                endOffset = doc_pages[num_page]["endOffset"]
                endOffset = min(endOffset, len(instring))
                instring = instring[:endOffset]
            if pred_mode:
                label = None
            elif args.task == "STATES" and args.multi:
                label = row[args.task].split(";")
            elif args.task == "DATE" and args.multi:
                label = row[args.task].split(" ")
            elif args.task == "DATE" and args.sub_date == "MONTH":
                label = row[args.task].split(" ")[0]
            elif args.task == "DATE" and args.sub_date == "YEAR":
                label = row[args.task].split(" ")[1]
            else:
                label = row[args.task]
            data_x.append(instring)
            data_y.append(label)
            data_ids.append((fileid, filename))
    return data_x, data_y, data_ids


def custom_vocabulary(args):
    if args.task == "AGENCY":
        data = pd.read_csv("agencies.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    elif args.task == "DOCTYPE":
        data = pd.read_csv("document_types.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    elif args.task == "STATES":
        data = pd.read_csv("states.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    elif args.task == "DATE":
        data = pd.read_csv("months.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")])) + list(map(str, range(1993, 2022)))
    else:
        return None


def custom_ner_vocabulary(args):
    if args.task == "AGENCY":
        data = pd.read_csv("orgs_ner.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    elif args.task == "STATES":
        data = pd.read_csv("locs_ner.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    elif args.task == "DATE":
        data = pd.read_csv("timex_ner.csv", encoding="utf8", comment='#')
        return list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                         for t in v.split(" ")]))
    else:
        return None


def calculate_positions(text, vocabulary):
    first_match = lambda x: -1 if x is None else x.start()
    positions = np.array([[first_match(re.search(r'\b%s\b' % v, t.lower()))
                           for v in vocabulary] for t in text])
    positions = np.where(positions == -1,
                         positions,
                         positions.max(axis=1).reshape(-1, 1) - positions)
    positions = rankdata(positions, method='min', axis=1)
    scaler = MinMaxScaler()
    positions = scaler.fit_transform(positions.transpose()).transpose()
    return positions


def train(args):
    assert not(args.task != "DATE" and args.regression)
    os.makedirs(args.save_model, exist_ok=True)
    train_df = pd.read_csv(args.train_file, encoding="utf8")
    train_text, train_y, _ = data(train_df, args)
    vocabulary = None
    if args.vocab:
        vocabulary = custom_vocabulary(args)
    elif args.ner:
        vocabulary = custom_ner_vocabulary(args)
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', vocabulary=vocabulary)
    train_x = vectorizer.fit_transform(train_text)
    if args.position or args.concat_position:
        train_positions = calculate_positions(train_text, vectorizer.vocabulary_.keys())
        train_x = np.concatenate((train_x.toarray(), train_positions), axis=1) if args.concat_position else train_positions
    if args.task == "STATES" and args.multi:
        model = MultiOutputClassifier(LinearSVC(max_iter=10000, verbose=1))
        mlb = MultiLabelBinarizer()
        train_y = mlb.fit_transform(train_y)
        dump(mlb, os.path.join(args.save_model, "multilabel_binarizer.joblib"))
    elif args.task == "DATE" and args.regression:
        model = LinearSVR(max_iter=10000, verbose=1)
    elif args.task == "DATE" and args.multi:
        model = MultiOutputClassifier(LinearSVC(max_iter=10000, verbose=1))
    else:
        model = LinearSVC(max_iter=10000, verbose=1)
    clf = model
    clf.fit(train_x, train_y)
    dump(vectorizer, os.path.join(args.save_model, "vectorizer.joblib"))
    dump(clf, os.path.join(args.save_model, "model.joblib"))
    if args.dev_file is not None:
        dev_df = pd.read_csv(args.dev_file, encoding="utf8")
        dev_text, dev_y, _ = data(dev_df, args)
        dev_x = vectorizer.transform(dev_text)
        if args.position or args.concat_position:
            dev_positions = calculate_positions(dev_text, vectorizer.vocabulary_.keys())
            dev_x = np.concatenate((dev_x.toarray(), dev_positions), axis=1) if args.concat_position else dev_positions
        if args.task == "STATES" and args.multi:
            dev_y = mlb.transform(dev_y)
        prediction = clf.predict(dev_x)
        if args.task == "DATE" and args.regression:
            print(mean_absolute_error(dev_y, prediction))
        elif args.task == "DATE"  and args.multi:
            dev_y = [" ".join(y) for y in dev_y]
            prediction = [" ".join(pred) for pred in prediction]
            print(precision_recall_fscore_support(dev_y, prediction, average="micro"))
        else:
            print(precision_recall_fscore_support(dev_y, prediction, average="micro"))


def predict(args):
    vectorizer = load(os.path.join(args.save_model, "vectorizer.joblib"))
    clf = load(os.path.join(args.save_model, "model.joblib"))
    input_df = pd.read_csv(args.input_file, encoding="utf8")
    input_text, _, input_ids = data(input_df, args, pred_mode=True)
    input_x = vectorizer.transform(input_text)
    if args.position or args.concat_position :
        input_positions = calculate_positions(input_text, vectorizer.vocabulary_.keys())
        input_x = np.concatenate((input_x.toarray(), input_positions), axis=1) if args.concat_position else input_positions
    predictions = clf.predict(input_x)
    if args.task == "STATES" and args.multi:
        mlb = load(os.path.join(args.save_model, "multilabel_binarizer.joblib"))
        rollback_predictions = multi_decision_function(clf, input_x)
        rollback_predictions = (rollback_predictions == rollback_predictions.max(axis=1)[:, None]).astype(int)
        zero_predictions = (predictions.sum(axis=1) == 0)
        predictions[zero_predictions, :] = rollback_predictions[zero_predictions, :]
        predictions = multi_decision_function(clf, input_x)
        predictions = [";".join(prediction) for prediction in mlb.inverse_transform(predictions)]
    elif args.task == "DATE" and args.multi:
        predictions = [" ".join(prediction) for prediction in predictions]
    ids, filenames = list(zip(*input_ids))
    df = pd.DataFrame(zip(ids, filenames, predictions), columns=["id", "filename", args.task])
    if args.task == "DATE" and args.regression:
        mean_date = load(os.path.join(args.save_model, "mean_date.joblib"))
        df = difference_to_date(df, mean_date)
    df.to_csv(args.out_file, encoding="utf8", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_dir', type=str,
                        help='Path to the directory containing json file')
    parser.add_argument('-train_file', type=str,
                        help='Path to the train csv file (e.g., train.csv).')
    parser.add_argument('-dev_file', type=str,
                        help='Path to the dev csv file (e.g., dev.csv).')
    parser.add_argument('-input_file', type=str,
                        help='Path to the input csv file to be predicted (e.g., input.csv).')
    parser.add_argument('-out_file', type=str,
                        help='Path to the output csv file with the predictions (e.g., output.csv).')
    parser.add_argument('-save_model', type=str,
                        help='Path to the directory where the model will be saved.')
    parser.add_argument('-task', type=str,
                        help='Task to be trained.')
    parser.add_argument('-volumes', type=int,
                        help='Use only the n first volumes of the documents.')
    parser.add_argument('-pages', type=int,
                        help='Use only the n first pages of the documents.')
    parser.add_argument('-alphanum', action="store_true",
                        help='Use simple alphanumeric sorting.')
    parser.add_argument('-random', action="store_true",
                        help='Use random sorting.')
    parser.add_argument('-regression', action="store_true",
                        help='Train for regression instead of classification.')
    parser.add_argument('-multi', action="store_true",
                        help='Train multi-task mode (months, years) for dates.')
    parser.add_argument('-vocab', action="store_true",
                        help='Use task customized vocabulary.')
    parser.add_argument('-ner', action="store_true",
                        help='Use ner based vocabulary.')
    parser.add_argument('-position', action="store_true",
                        help='Use first position vector.')
    parser.add_argument('-concat_position', action="store_true",
                        help='Attach first position vector.')
    parser.add_argument('-trunc', type=int,
                        help='Truncate the input text.')
    parser.add_argument('-sub_date', type=str,
                        help='Train just MONTH or YEAR.')
    args = parser.parse_args()
    if args.input_file is not None:
        predict(args)
    else:
        train(args)
