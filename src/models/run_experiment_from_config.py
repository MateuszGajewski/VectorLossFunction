import configparser
import time

import mlflow
import torch

import src.data.datasets as dataset
import src.models.data_loaders as data_loaders
import src.models.loss_function as loss_function
from pathlib import Path
import torch.optim as optim
import torchmetrics
from datetime import date


class Experiment:
    def __init__(self, config_file_path: Path):
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.cls = None
        self.optimizer = None
        self.device = None
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        mlflow.set_tracking_uri('../../mlruns')
        mlflow.set_experiment(self.config['training']['experiment'])

    def save_model(self, model_ft):
        mlflow.pytorch.log_model(model_ft, str(model_ft))
        # mlflow.pytorch.save_model(model_ft, '../../' + str(date.today()) + '/')
        #   mlflow.log_metric('history',hist)
        # torch.save(model_ft.state_dict(),
        #           './' + str(date.today()) + '.pth')


    def build_objects(self):
        self.device = self.config['training']['device']
        self.criterion = eval(self.config['training']['loss_function'])()

        data_loader = eval(self.config['data']['data_loader'])

        data_loader = data_loader()
        self.train_loader, self.test_loader, self.cls = data_loader.get_data_loaders(self.config)
        self.optimizer = eval(self.config['training']['optimizer'])(self.cls.parameters(),
                                                                    lr=float(self.config['training']['lr']),
                                                                    momentum=float(self.config['training']['momentum']))

    def train_and_log_model(self):
        with mlflow.start_run() as run:
            mlflow.log_param('train_dataset', self.config['data']['train_data'])
            mlflow.log_param('test_dataset', self.config['data']['test_data'])
            mlflow.log_param('model name', self.config['training']['classifier'])
            mlflow.log_param('number of dimensions', self.config['training']['out_dim'])
            mlflow.log_param('Batch size', self.config['training']['batch_size'])
            mlflow.log_param('epochs', self.config['training']['epochs'])
            mlflow.log_param('learning_rate', self.config['training']['lr'])
            mlflow.log_param('momentum', self.config['training']['momentum'])
            self.train()
            self.validate()
            self.save_model(self.cls)

    def train(self):
        self.cls.fit(self.config, self.optimizer, self.criterion, self.train_loader)

    def validate(self):
        self.cls.validate(self.config, self.optimizer, self.criterion, self.test_loader)


if __name__ == "__main__":
    experiment = Experiment(Path('./configs/simple_table_config.ini'))
    experiment.build_objects()
    experiment.train_and_log_model()
