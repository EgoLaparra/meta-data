import os
import random
import numpy as np
import torch
from torch import nn
from joblib import dump, load
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import (Trainer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers import set_seed
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from datasets import Dataset


class TransformersModel:

    def __init__(self, model_name, setting):
        self.model_name = model_name
        self.epochs = setting.epochs
        self.batch_size = setting.batch_size
        self.lr = setting.lr
        self.seed_val = setting.seed_val
        self.save_path = None
        self.label_binarizer = None
        self.config = None
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.trainer = None
        self.data_collator = None


    def initialize_model_train(self, model, save_path, labels):
        if self.seed_val is not None:
            set_seed(self.seed_val)
        self.save_path = save_path
        self.label_binarizer = LabelBinarizer()
        self.config = AutoConfig.from_pretrained(model)
        self.config.num_labels = len(labels) + 1
        self.config.id2label = {e: label for e, label in enumerate(["UNK"] + labels)}
        self.config.label2id = {label: e for e, label in enumerate(["UNK"] + labels)}
        self.tokenizer = AutoTokenizer.from_pretrained(model, config=self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=self.config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<VOCAB>']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.training_args = TrainingArguments(self.save_path,
                                               num_train_epochs=self.epochs,
                                               per_device_train_batch_size=self.batch_size,
                                               learning_rate=self.lr,
                                               evaluation_strategy="epoch",
                                               load_best_model_at_end=True,
                                               metric_for_best_model="eval_f1",
                                               save_strategy="epoch")

    def initialize_model_predict(self, model_path):
        self.label_binarizer = load(os.path.join(model_path, "label_binarizer.joblib"))
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)

    def prepare_label(self, label):
        if hasattr(self.label_binarizer, 'classes_'):
            one_hot = self.label_binarizer.transform(label)
            return {'label': one_hot.argmax(axis=1) + one_hot.sum(axis=1)}
        else:
            one_hot = self.label_binarizer.fit_transform(label)
            return {'label': one_hot.argmax(axis=1) + one_hot.sum(axis=1)}

    def prepare_dataset(self, dataset, shuffle, skip_labels):
        prepared_dataset = Dataset.from_pandas(dataset.df)
        prepared_dataset = prepared_dataset.map(lambda examples:
                                                self.tokenizer(examples['text'],
                                                               padding=True, truncation=True,
                                                               max_length=512),
                                                batched=True,
                                                batch_size=100)
        if not skip_labels:
            prepared_dataset = prepared_dataset.map(lambda examples:
                                                    self.prepare_label(examples['label']),
                                                    batched=True,
                                                    batch_size=len(prepared_dataset))
        if shuffle:
            prepared_dataset.shuffle(seed=5)
        return prepared_dataset

    def train(self, train_dataset, dev_dataset, save_path, resume_checkpoint=False):
        self.initialize_model_train(self.model_name, save_path, train_dataset.labels)
        if train_dataset.dataset is not None:
            train_prepared_dataset = train_dataset.dataset
            train_prepared_dataset = train_prepared_dataset.map(lambda examples:
                                                                self.prepare_label(examples['label']),
                                                                batched=True,
                                                                batch_size=len(train_prepared_dataset))
            train_prepared_dataset.shuffle(seed=5)
        else:
            train_prepared_dataset = self.prepare_dataset(train_dataset, True, False)
        train_prepared_dataset.save_to_disk(os.path.join(save_path, "train_dataset"))        
        if dev_dataset.dataset is not None:
            dev_prepared_dataset = dev_dataset.dataset
            dev_prepared_dataset = dev_prepared_dataset.map(lambda examples:
                                                            self.prepare_label(examples['label']),
                                                            batched=True,
                                                            batch_size=len(dev_prepared_dataset))
        else:
            dev_prepared_dataset = self.prepare_dataset(dev_dataset, False, False)
        dev_prepared_dataset.save_to_disk(os.path.join(save_path, "dev_dataset"))
        self.trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=train_prepared_dataset,
            eval_dataset=dev_prepared_dataset,
            compute_metrics=self.compute_metrics
        )
        print(self.trainer.model.device)
        self.trainer.train(resume_checkpoint)
        self.save()

    def predict(self, input_dataset, model_path):
        self.initialize_model_predict(model_path)
        if input_dataset.dataset is not None:
            input_prepared_dataset = input_dataset.dataset
        else:
            input_prepared_dataset = self.prepare_dataset(input_dataset, False, True)
        input_prepared_dataset.save_to_disk(os.path.join(model_path, "input_dataset"))
        self.trainer = Trainer(
            self.model
        )
        prediction = self.trainer.predict(input_prepared_dataset).predictions
        prediction = self.prepare_prediction(prediction)
        return self.label_binarizer.inverse_transform(prediction)

    def save(self):
        self.trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        dump(self.label_binarizer, os.path.join(self.save_path, "label_binarizer.joblib"))

    @staticmethod
    def prepare_prediction(prediction):
        one_hot = np.zeros(prediction.shape)
        one_hot[np.arange(one_hot.shape[0]), prediction.argmax(axis=1)] = 1
        return np.column_stack((one_hot, np.zeros((one_hot.shape[0], 1))))[:, 1:-1]

    @staticmethod
    def compute_metrics(eval_prediction):
        labels = eval_prediction.label_ids
        predictions = eval_prediction.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class TransformersMultiLabel(TransformersModel):

    def initialize_model_train(self, model, save_path, labels):
        if self.seed_val is not None:
            # self.set_seed(5)
            # set_seed(5)
            set_seed(self.seed_val)
        self.save_path = save_path
        self.label_binarizer = MultiLabelBinarizer()
        self.config = AutoConfig.from_pretrained(model)
        self.config.num_labels = len(labels)
        self.config.id2label = {e: label for e, label in enumerate(labels)}
        self.config.label2id = {label: e for e, label in enumerate(labels)}
        self.config.problem_type = 'multi_label_classification'
        self.tokenizer = AutoTokenizer.from_pretrained(model, config=self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=self.config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<VOCAB>']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.training_args = TrainingArguments(self.save_path,
                                               num_train_epochs=self.epochs,
                                               per_device_train_batch_size=self.batch_size,
                                               learning_rate=self.lr,
                                               evaluation_strategy="epoch",
                                               load_best_model_at_end=True,
                                               metric_for_best_model="eval_f1",
                                               save_strategy="epoch")

    def prepare_label(self, label):
        if hasattr(self.label_binarizer, 'classes_'):
            return {'label': self.label_binarizer.transform(label).astype(float)}
        else:
            return {'label': self.label_binarizer.fit_transform(label).astype(float)}

    @staticmethod
    def prepare_prediction(prediction):
        multilabel_one_hot = np.zeros(prediction.shape)
        multilabel_one_hot[np.where(prediction >= 0)] = 1
        none_output = np.argwhere(multilabel_one_hot.sum(axis=1) == 0)
        rollback_one_hot = np.zeros(prediction.shape)
        rollback_one_hot[np.arange(rollback_one_hot.shape[0]), np.argmax(prediction, axis=1)] = 1
        multilabel_one_hot[none_output] = rollback_one_hot[none_output]
        return multilabel_one_hot

    @staticmethod
    def compute_metrics(eval_prediction):
        labels = eval_prediction.label_ids
        predictions = np.zeros(eval_prediction.predictions.shape)
        predictions[np.where(eval_prediction.predictions >= 0)] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class TransformersMultiOutput(TransformersModel):

    def __init__(self, model_name, setting):
        super().__init__(model_name, setting)
        self.label_binarizer_1 = None
        self.label_binarizer_2 = None

    def initialize_model_train(self, model_path, save_path, labels):
        if self.seed_val is None:
            # self.set_seed(5)
            # set_seed(5)
            set_seed(self.seed_val)
        self.save_path = save_path
        self.label_binarizer_1 = LabelBinarizer()
        self.label_binarizer_2 = LabelBinarizer()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.num_labels_1 = len(labels[0]) + 1
        self.config.id2label_1 = {e: label for e, label in enumerate(["UNK"] + labels[0])}
        self.config.label2id_1 = {label: e for e, label in enumerate(["UNK"] + labels[0])}
        self.config.num_labels_2 = len(labels[1]) + 1
        self.config.id2label_2 = {e: label for e, label in enumerate(["UNK"] + labels[1])}
        self.config.label2id_2 = {label: e for e, label in enumerate(["UNK"] + labels[1])}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForMultiOutputClassification.from_pretrained(model_path, config=self.config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<VOCAB>']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.training_args = TrainingArguments(self.save_path,
                                               num_train_epochs=self.epochs,
                                               per_device_train_batch_size=self.batch_size,
                                               evaluation_strategy="epoch",
                                               load_best_model_at_end=True,
                                               metric_for_best_model="eval_f1_mean",
                                               save_strategy="epoch")

    def initialize_model_predict(self, model_path):
        self.label_binarizer_1 = load(os.path.join(model_path, "label_binarizer_1.joblib"))
        self.label_binarizer_2 = load(os.path.join(model_path, "label_binarizer_2.joblib"))
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForMultiOutputClassification.from_pretrained(model_path, config=self.config)

    def prepare_label(self, label):
        if hasattr(self.label_binarizer, 'classes_'):
            one_hot_1 = self.label_binarizer_1.transform(np.array(label)[:, 0])
            one_hot_2 = self.label_binarizer_2.transform(np.array(label)[:, 1])
            return {"label": np.stack((one_hot_1.argmax(axis=1) + one_hot_1.sum(axis=1),
                                       one_hot_2.argmax(axis=1) + one_hot_2.sum(axis=1)),
                                      axis=1)}
        else:
            one_hot_1 = self.label_binarizer_1.fit_transform(np.array(label)[:, 0])
            one_hot_2 = self.label_binarizer_2.fit_transform(np.array(label)[:, 1])
            return {"label": np.stack((one_hot_1.argmax(axis=1) + one_hot_1.sum(axis=1),
                                       one_hot_2.argmax(axis=1) + one_hot_2.sum(axis=1)),
                                      axis=1)}

    def predict(self, input_dataset, model_path):
        self.initialize_model_predict(model_path)
        if input_dataset.dataset is not None:
            input_prepared_dataset = input_dataset.dataset
        else:
            input_prepared_dataset = self.prepare_dataset(input_dataset, False, True)
        self.trainer = Trainer(
            self.model
        )
        prediction = self.trainer.predict(input_prepared_dataset).predictions
        prediction_1 = self.prepare_prediction(prediction[0])
        prediction_1 = self.label_binarizer_1.inverse_transform(prediction_1)
        prediction_2 = self.prepare_prediction(prediction[1])
        prediction_2 = self.label_binarizer_2.inverse_transform(prediction_2)
        return np.stack((prediction_1, prediction_2), axis=1)

    def save(self):
        self.trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        dump(self.label_binarizer_1, os.path.join(self.save_path, "label_binarizer_1.joblib"))
        dump(self.label_binarizer_2, os.path.join(self.save_path, "label_binarizer_2.joblib"))

    @staticmethod
    def compute_metrics(eval_prediction):
        labels = eval_prediction.label_ids
        predictions_1 = eval_prediction.predictions[0].argmax(-1)
        precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(labels[:, 0], predictions_1, average='micro')
        predictions_2 = eval_prediction.predictions[1].argmax(-1)
        precision_2, recall_2, f1_2, _ = precision_recall_fscore_support(labels[:, 1], predictions_2, average='micro')
        return {
            'f1_1': f1_1,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'f1_2': f1_2,
            'precision_2': precision_2,
            'recall_2': recall_2,
            'f1_mean': (f1_1 + f1_2) / 2
        }


@dataclass
class MultiOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_1: torch.FloatTensor = None
    logits_2: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AutoModelForMultiOutputClassification(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.base_model_prefix = "multioutput"
        self.num_labels_1 = config.num_labels_1
        self.num_labels_2 = config.num_labels_2
        self.alpha = 1.5
        self.config = config
        self.add_module(config.model_type, AutoModel.from_config(config))
        self.linear_1 = nn.Linear(768, 768, bias=True)
        self.dropout_1 = nn.Dropout(0.1)
        self.classifier_1 = nn.Linear(768, self.num_labels_1, bias=True)
        self.linear_2 = nn.Linear(768, 768, bias=True)
        self.dropout_2 = nn.Dropout(0.1)
        self.classifier_2 = nn.Linear(768, self.num_labels_2, bias=True)
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_input_embeddings(self) -> nn.Module:
        base_model = getattr(self, self.config.model_type, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        base_model = getattr(self, self.config.model_type, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        encoder = getattr(self, self.config.model_type)
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        linear_output_1 = self.linear_1(encoder_outputs[0])
        sequence_output_1 = self.dropout_1(linear_output_1)
        logits_1 = self.classifier_1(sequence_output_1[:, 0, :].view(-1, 768))
        linear_output_2 = self.linear_2(encoder_outputs[0])
        sequence_output_2 = self.dropout_2(linear_output_2)
        logits_2 = self.classifier_2(sequence_output_2[:, 0, :].view(-1, 768))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_1 = loss_fct(logits_1.view(-1, self.num_labels_1), labels[:, 0].view(-1).long())
            loss_2 = loss_fct(logits_2.view(-1, self.num_labels_2), labels[:, 1].view(-1).long())
            loss = loss_1 + self.alpha * loss_2

        return MultiOutput(loss=loss, logits_1=logits_1, logits_2=logits_2,
                           hidden_states=encoder_outputs.hidden_states,
                           attentions=encoder_outputs.attentions)

