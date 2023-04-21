import argparse
import regex
from pandas import read_csv
from datetime import datetime
from dateutil.relativedelta import relativedelta


def count_matches(x, task, match_function):
    matches = 0
    gold_column = task + "_x"
    pred_column = task + "_y"
    for prediction in set(x[pred_column]):
        for gold in set(x[gold_column]):
            matches += match_function(gold, prediction)
    if matches == 0:
        print("%s\t%s\t%s\t%s" % (x["filename"], task, x[gold_column], x[pred_column]))
    elif task == "DATE":  # There can be more than 1 date correct
        matches = 1
    return matches


def simple_match(gold, prediction):
    if prediction == gold:
        return 1
    else:
        return 0


def norm_date(date):
    if date is None:
        return date
    date = regex.sub(r'([A-Za-z]+)[^A-Za-z0-9]+([0-9]+)', r'\1 \2', date)
    return date


def date_match(gold, prediction):
    gold = datetime.strptime(gold, '%B %Y')
    try:
        prediction = norm_date(prediction)
        prediction = datetime.strptime(prediction, '%B %Y')
    except ValueError:
        return 0
    if gold - relativedelta(months=3) <= prediction <= gold:
        return 1
    else:
        return 0


def run_task(merged, task, match_function=simple_match):
    gold_column = task + "_x"
    pred_column = task + "_y"
    merged = merged[["id", "filename", gold_column, pred_column]]
    merged = merged[~merged[gold_column].isna()]
    merged[gold_column] = merged[gold_column].str.split(";")
    merged[pred_column] = merged[pred_column].str.split(";")
    total_gold = merged[gold_column].apply(lambda x: len(x)).sum()
    total_pred = merged[pred_column].dropna().apply(lambda x: len(x)).sum()
    total_match = merged.dropna().apply(lambda x: count_matches(x, task, match_function), axis=1).sum()
    precision = total_match / total_pred
    recall = total_match / total_gold
    f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score


def run(args):
    gold = read_csv(args.gold)
    pred = read_csv(args.pred, engine="python")
    pred = pred.rename(columns={"fileid": "id"})
    if args.task == "STATES" or args.task is None:
        pred["STATES"] = pred["STATES"].str.replace("(","").str.replace(")","").str.replace("'","").str.replace(" ","").str.replace(r",$","").str.split(",").str.join(";")
    merged = gold.merge(pred, on=["id", "filename"])
    tasks = [("DOCTYPE", simple_match), ("DATE", date_match),
             ("STATES", simple_match), ("AGENCY", simple_match)]
    scores = {}
    for task, match_function in tasks:
        if args.task is None or args.task == task:
            scores[task] = run_task(merged, task, match_function=match_function)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold', type=str,
                        help='Gold file.')
    parser.add_argument('-pred', type=str,
                        help='Prediction file')
    parser.add_argument('-task', type=str,
                        help='Run scorer only for this task')
    args = parser.parse_args()
    scores = run(args)
    for task in scores:
        print("=== %s ===" % task)
        print("P: %0.3f\tR: %0.3f\tF1: %0.3f" % (scores[task][0], scores[task][1], scores[task][2]))
