import configparser
import time

import mlflow
import torch

import src.data.datasets as dataset
import src.models.loss_function as loss_function
import src.models.classifiers as classifiers
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchmetrics
from datetime import date


class Experiment:
    def __init__(self, config_file_pat: Path):
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.cls = None
        self.optimizer = None
        self.device = None
        self.config = configparser.ConfigParser()
        self.config.read('./configs/config.ini')
        mlflow.set_tracking_uri('../../mlruns')
        mlflow.set_experiment(self.config['training']['experiment'])

    def save_model(self, model_ft):
        mlflow.pytorch.log_model(model_ft, "models")
        #mlflow.pytorch.save_model(model_ft, '../../' + str(date.today()) + '/')
        #   mlflow.log_metric('history',hist)
        #torch.save(model_ft.state_dict(),
        #           './' + str(date.today()) + '.pth')

    def build_objects(self):
        dataset = eval(self.config['data']['dataset'])
        transform = transforms.Compose(  # composing several transforms together
            [transforms.ToTensor(),  # to tensor object
             transforms.Normalize((float(self.config['data']['mean'])), (float(self.config['data']['mean'])))])
        dataset_train = dataset(Path(self.config['data']['train_data']), transform=transform,
                                label_to_vec_function=simple_f)
        dataset_test = dataset(Path(self.config['data']['test_data']), transform=transform,
                               label_to_vec_function=simple_f)

        self.train_loader = DataLoader(dataset_train, batch_size=int(self.config['training']['batch_size']),
                                       shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(dataset_test, batch_size=int(self.config['training']['batch_size']),
                                      shuffle=True, pin_memory=True)
        self.device = self.config['training']['device']
        self.criterion = eval(self.config['training']['loss_function'])()
        self.cls = eval(self.config['training']['classifier'])(int(self.config['training']['out_dim'])).to(self.device)
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
        since = time.time()
        for epoch in range(int(self.config['training']['epochs'])):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.cls(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')
        time_elapsed = time.time() - since
        mlflow.log_metric('Training time', time_elapsed)

    def validate(self):
        total_loss = 0.0
        total_number = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.cls(inputs)
                outputs = outputs.to(self.device)
                loss = self.criterion(outputs, labels)
                self.optimizer.step()
                total_loss += loss.item()
                total_number += labels.size(0)

        print(f'Finished validaion, avg loss: {total_loss / total_number}')
        mlflow.log_metric('Avg test loss', total_loss/total_number)



def simple_f(l, lh):
    return torch.tensor([l, lh])


if __name__ == "__main__":
    experiment = Experiment(Path("./configs/config.ini"))
    experiment.build_objects()
    experiment.train_and_log_model()
