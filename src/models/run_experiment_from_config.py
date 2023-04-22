import configparser
import time
from datetime import date
from pathlib import Path
import ast
import mlflow
import json
import torch
import torch.optim as optim
import torchmetrics
import src.models.metrics as metrics
import src.data.datasets as dataset
import src.models.data_loaders as data_loaders
import src.models.loss_function as loss_function
import src.models.vector_to_label_transformers as vector_to_label_transformers


class Experiment:
    def __init__(self, config_file_path: Path):
        self.vector_to_label_transformer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.cls = None
        self.optimizer = None
        self.device = None
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        self.label_hierarchy = None
        self.metrics = {}
        mlflow.set_tracking_uri("../../mlruns")
        mlflow.set_experiment(self.config["training"]["experiment"])

    def save_model(self, model_ft):
        mlflow.pytorch.log_model(model_ft, str(model_ft))

    def build_metrics(self):
        if self.config.has_option('inference', 'metrics'):
            for i in ast.literal_eval(self.config.get("inference", "metrics")):
                if self.label_hierarchy:
                    self.metrics[i] = eval(i)(self.label_hierarchy, self.device)

    def build_objects(self):
        self.device = self.config["training"]["device"]

        if self.config.has_option('data', 'labels_hierarchy'):
            self.label_hierarchy = json.load(open(self.config['data']['labels_hierarchy']))
            self.criterion = eval(self.config["training"]["loss_function"])(self.device,
                                                                            self.label_hierarchy)
        else:
            self.criterion = eval(self.config["training"]["loss_function"])(self.device)
        if self.config.has_option('training', 'recalculate_period'):
            self.criterion.set_recalculate_period(self.config['training']['recalculate_period'])

        self.vector_to_label_transformer = eval(self.config
                                                ["inference"]["vector_to_label_transformer"])()

        data_loader = eval(self.config["data"]["data_loader"])

        data_loader = data_loader()
        self.train_loader, self.test_loader, self.cls = data_loader.get_data_loaders(
            self.config
        )
        self.optimizer = eval(self.config["training"]["optimizer"])(
            self.cls.parameters(),
            lr=float(self.config["training"]["lr"]),
            momentum=float(self.config["training"]["momentum"]),
        )

    def train_and_log_model(self):
        mlflow.log_param("train_dataset", self.config["data"]["train_data"])
        mlflow.log_param("test_dataset", self.config["data"]["test_data"])
        mlflow.log_param("model name", self.config["training"]["classifier"])
        mlflow.log_param("number of dimensions", self.config["training"]["out_dim"])
        mlflow.log_param("Batch size", self.config["training"]["batch_size"])
        mlflow.log_param("epochs", self.config["training"]["epochs"])
        mlflow.log_param("learning_rate", self.config["training"]["lr"])
        mlflow.log_param("momentum", self.config["training"]["momentum"])
        mlflow.log_param("loss_functon", self.config['training']['loss_function'])
        mlflow.log_param("vector_to_label_transformer",
                         self.config["inference"]["vector_to_label_transformer"])
        if self.config.has_option('training', 'recalculate_period'):
            mlflow.log_param( 'recalculate_period' ,self.config['training']['recalculate_period'])
        self.train()
        self.validate()
        self.save_model(self.cls)

    def train(self):
        self.cls.fit(self.config, self.optimizer, self.criterion, self.train_loader,
                     self.vector_to_label_transformer)

    def validate(self):
        self.cls.validate(self.config, self.test_loader, self.metrics,
                          self.vector_to_label_transformer)


if __name__ == "__main__":
    experiment = Experiment(Path("./configs/visual_config_davies_bouldin.ini"))
    experiment.build_objects()
    experiment.build_metrics()
    experiment.train_and_log_model()
