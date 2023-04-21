import argparse

from models import *
from data_sets import *


def init_model(args, setting):
    elif args.task == "DATE" and args.multi:
        return TransformersMultiOutput(args.model_name, setting)
    else:
        return TransformersModel(args.model_name, setting)

def init_dataset(file_path, args, setting):
    if args.task == "STATES":
        return TransformersDatasetStates.from_csv(file_path, args.json_dir, setting)
    elif args.task == "DATE":
        return TransformersDatasetDate.from_csv(file_path, args.json_dir, setting)
    elif args.task == "DOCTYPE":
        return TransformersDatasetDocType.from_csv(file_path, args.json_dir, setting)
    elif args.task == "AGENCY":
        return TransformersDatasetAgency.from_csv(file_path, args.json_dir, setting)
    else:
        raise Exception("Wrong task!")

def load_dataset(dataset_path, args, setting):
    if args.task == "STATES":
        return TransformersDatasetStates.from_dataset(dataset_path, setting)
    elif args.task == "DATE":
        return TransformersDatasetDate.from_dataset(dataset_path, setting)
    elif args.task == "DOCTYPE":
        return TransformersDatasetDocType.from_dataset(dataset_path, setting)
    elif args.task == "AGENCY":
        return TransformersDatasetAgency.from_dataset(dataset_path, setting)
    else:
        raise Exception("Wrong task!")


def run(args):
    if args.input_file is not None:
        setting = Setting(args, pred_mode=True)
        model = init_model(args, setting)
        if args.input_dataset is not None:
            input_dataset = load_dataset(args.input_dataset, args, setting)
        else:
            input_dataset = init_dataset(args.input_file, args, setting)
        input_dataset.predictions = model.predict(input_dataset, args.save_path)
        input_dataset.write_csv(args.out_file)
    else:
        setting = Setting(args)
        model = init_model(args, setting)
        if args.train_dataset is not None:
            train_dataset = load_dataset(args.train_dataset, args, setting)
        else:
            train_dataset = init_dataset(args.train_file, args, setting)
        if args.dev_dataset is not None:
            dev_dataset = load_dataset(args.dev_dataset, args, setting)
        else:
            dev_dataset = init_dataset(args.dev_file, args, setting)
        print("TRAIN: " + str(args.seed_val))
        model.train(train_dataset, dev_dataset, args.save_path, resume_checkpoint=args.resume_checkpoint)


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
    parser.add_argument('-save_path', type=str,
                        help='Path to the directory where the model will be saved.')
    parser.add_argument('-model_name', type=str, default='roberta-base',
                        help='Name of the pre-trained model to use (default: roberta-base).')
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
    parser.add_argument('-mark', action="store_true",
                        help='Mark tokens from regex with <VOCAB> in the input text.')
    parser.add_argument('-sub_date', type=str,
                        help='Train just MONTH or YEAR.')
    parser.add_argument('-text_split', type=int, default=None,
                        help='Split text into substring of this size.')
    parser.add_argument('-train_dataset', type=str, default=None,
                        help='Path to a tokenized train dataset.')
    parser.add_argument('-dev_dataset', type=str, default=None,
                        help='Path to a tokenized dev dataset.')
    parser.add_argument('-input_dataset', type=str, default=None,
                        help='Path to a tokenized input dataset.')
    parser.add_argument('-ner_dir', type=str, default=None,
                        help='Path to directory with ner annotation.')
    parser.add_argument('-all_volumes', action="store_true",
                        help='Use all volumes.')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-seed_val', type=int)
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-resume_checkpoint', action="store_true")

    args = parser.parse_args()
    run(args)
