import os
import json
import random
import regex as re
import pandas as pd
import numpy as np
import pickle as pkl
from datasets import load_from_disk, concatenate_datasets

from rulebased_sort import sort


class Setting:

    def __init__(self, args, pred_mode=False):
        self.pred_mode = pred_mode
        self.regression = args.regression
        self.multi = args.multi
        self.alphanum = args.alphanum
        self.random = args.random
        self.volumes = args.volumes
        self.pages = args.pages
        self.mark = args.mark
        self.sub_date = args.sub_date
        self.text_split = args.text_split
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.seed_val = args.seed_val
        self.ner_dir = args.ner_dir
        self.all_volumes = args.all_volumes

        
class TransformersDataset:

    def __init__(self, task):
        self.task = task
        self.df = pd.DataFrame(columns=['fileid', 'filename', 'volid', 'text', 'label'])
        self.labels = []
        self.predictions = None
        self.ref_point = None
        self.dataset = None

    def add_example(self, text, label, fileid, filename, volid):
        self.df = self.df.append({'fileid': fileid, 'filename': filename, 'volid': volid,
                                  'text': text, 'label': label},
                                 ignore_index=True)

    def update_label_set(self, label):
        if label not in self.labels:
            self.labels.append(label)

    def mark_text(self, text):
        vocab = None
        if self.task == "AGENCY":
            data = pd.read_csv("agencies.csv", encoding="utf8", comment='#')
            vocab =  list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                               for t in v.split(" ")]))
        elif self.task == "DOCTYPE":
            data = pd.read_csv("document_types.csv", encoding="utf8", comment='#')
            vocab = list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                              for t in v.split(" ")]))
        elif self.task == "STATES":
            data = pd.read_csv("states.csv", encoding="utf8", comment='#')
            vocab = list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                              for t in v.split(" ")]))
        elif self.task == "DATE":
            data = pd.read_csv("months.csv", encoding="utf8", comment='#')
            vocab = list(set([t.lower() for v in data.fillna("").values.flatten() if v != ""
                              for t in v.split(" ")])) + list(map(str, range(1993, 2022)))
        if vocab is None:
            return text
        else:
            return re.sub(r'\b(%s)\b' % "|".join(vocab), r'<VOCAB> \1 <VOCAB>', text, flags=re.IGNORECASE)

    def mark_ner(self, text, filename):
        return text
        
    def load_dataset(self, dataset_path, setting):
        # splits = []
        # for split in os.listdir(dataset_path):
        #     split_path = os.path.join(dataset_path, split)
        #     splits.append(
        #         load_from_disk(split_path)
        #     )
        # self.dataset = concatenate_datasets(splits)
        self.dataset = load_from_disk(dataset_path)
        if not setting.pred_mode:
            self.dataset = self.dataset.filter(lambda row: row[self.task] != "")
        if not setting.pred_mode:
            self.dataset = self.dataset.map(lambda row: {"label": self.get_label(row[self.task], setting)}, batched=False)
            if type(self.dataset['label'][0]) == list:
                self.labels = list(set([label for labels in self.dataset['label'] for label in labels]))
            else:
                self.labels = list(set(self.dataset['label']))

    def load_csv(self, csv_path, json_dir, setting):
        df = pd.read_csv(csv_path, encoding="utf8")
        if not setting.pred_mode:
            df = df[~df[self.task].isna()]
        for row in df.iterrows():
            row = row[1].fillna("")
            fileid = row["id"]
            filename = row["filename"]
            json_path = os.path.join(json_dir, filename + ".json")
            texts = self.from_json(json_path, setting)
            for e, text in enumerate(texts):
                if setting.mark:
                    text = self.mark_ner(text, filename) if setting.ner_dir is not None else self.mark_text(text)
                label = self.get_label(row[self.task], setting)
                if not setting.pred_mode:
                    self.update_label_set(label)
                if setting.text_split is not None:
                    for i in range(0, len(text), setting.text_split):
                        self.add_example(
                            text[i: i+setting.text_split], label, fileid, filename, e
                        )
                else:
                    self.add_example(
                        text, label, fileid, filename, e
                    )


    def write_csv(self, output_path):
        df = self.df[["fileid", "filename"]]
        df[self.task] = self.predictions
        df.to_csv(output_path, encoding="utf8", index=False)

    @staticmethod
    def get_label(label, setting):
        if setting.pred_mode:
            return None
        return label

    @staticmethod
    def from_json(json_path, setting):
        assert os.path.exists(json_path)
        with open(json_path, encoding="ascii", errors="ignore") as json_file:
            json_data = json.load(json_file)
            volumes = json_data["volumes"]
            if setting.alphanum:
                volumes = sorted(volumes, key=lambda x: x["name"])
            elif setting.random:
                random.seed(0)
                random.shuffle(volumes)
            else:
                volumes = sort(volumes, key=lambda x: x["name"])
            if setting.volumes is not None:
                num_volumes = min(setting.volumes, len(volumes))
                volumes = volumes[:num_volumes]
            texts = []
            for volume in volumes:
                text = volume["text"]
                if setting.pages is not None:
                    doc_pages = [page for volume in json_data["volumes"] for page in volume["pages"]]
                    num_page = min(setting.pages, len(doc_pages)) - 1
                    endOffset = doc_pages[num_page]["endOffset"]
                    endOffset = min(endOffset, len(text))
                    text = text[:endOffset]
                texts.append(text)
            if not setting.all_volumes:
                texts = ["".join(texts for volume in volumes)]
        return texts


class TransformersDatasetAgency(TransformersDataset):

    @classmethod
    def from_csv(cls, csv_path, json_dir, setting):
        dataset = cls("AGENCY")
        dataset.load_csv(csv_path, json_dir, setting)
        return dataset

    @classmethod
    def from_dataset(cls, dataset_path, setting):
        dataset = cls("AGENCY")
        dataset.load_dataset(dataset_path, setting)
        return dataset


class TransformersDatasetDocType(TransformersDataset):

    @classmethod
    def from_csv(cls, csv_path, json_dir, setting):
        dataset = cls("DOCTYPE")
        dataset.load_csv(csv_path, json_dir, setting)
        return dataset

    @classmethod
    def from_dataset(cls, dataset_path, setting):
        dataset = cls("DOCTYPE")
        dataset.load_dataset(dataset_path, setting)
        return dataset


class TransformersDatasetDate(TransformersDataset):

    def update_label_set(self, label):
        if len(self.labels) == 2 and isinstance(self.labels[0], list):
            if label[0] not in self.labels[0]:
                self.labels[0].append(label[0])
            if label[1] not in self.labels[1]:
                self.labels[1].append(label[1])
        elif label not in self.labels:
            self.labels.append(label)

    def write_csv(self, output_path):
        df = self.df[["fileid", "filename"]]
        if np.ndim(self.predictions) == 2:
            df[self.task] = list(map(" ".join, self.predictions))
        else:
            df[self.task] = self.predictions
        df.to_csv(output_path, encoding="utf8", index=False)

    @staticmethod
    def get_label(label, setting):
        if setting.pred_mode:
            return None
        if setting.multi:
            return label.split(" ")
        elif setting.sub_date == "MONTH":
            return label.split(" ")[0]
        elif setting.sub_date == "YEAR":
            return label.split(" ")[1]
        else:
            return label

    @classmethod
    def from_csv(cls, csv_path, json_dir, setting):
        dataset = cls("DATE")
        if setting.multi:
            dataset.labels = [[], []]
        dataset.load_csv(csv_path, json_dir, setting)
        return dataset

    @classmethod
    def from_dataset(cls, dataset_path, setting):
        dataset = cls("DATE")
        if setting.multi:
            dataset.labels = [[], []]
        dataset.load_dataset(dataset_path, setting)
        return dataset


class TransformersDatasetStates(TransformersDataset):

    def update_label_set(self, label):
        for l in label:
            if l not in self.labels:
                self.labels.append(l)

    @staticmethod
    def get_label(label, setting):
        if setting.pred_mode:
            return None
        return label.split(";")

    @classmethod
    def from_csv(cls, csv_path, json_dir, setting):
        dataset = cls("STATES")
        dataset.load_csv(csv_path, json_dir, setting)
        return dataset

    @classmethod
    def from_dataset(cls, dataset_path, setting):
        dataset = cls("STATES")
        dataset.load_dataset(dataset_path, setting)
        return dataset
