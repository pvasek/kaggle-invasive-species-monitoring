import os
import time
import csv
import torch
from torch import optim, nn, min, max
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from image_folder import ImageFolder2
from tensorboard_monitor import TensorboardMonitor
import json
from sklearn import metrics

batch_size = 10
no_workers = 4
use_gpu = torch.cuda.is_available()
    
class Model(object):    
    def __init__(self, 
        root_directory: str,
        log_root_directory: str,
        model: nn.Module,
        name: str,        
        lr = 0.001,
        momentum = 0.9,
        gamma = 0.1,
        step_size = 10) -> None:                

        """"
        model: model instance (models.resnet152(pretrained=True))
        """

        if hasattr(model, 'fc'):
            print('reseting last layer...')
            number_of_features = model.fc.in_features
            model.fc = nn.Linear(number_of_features, 2)

        if use_gpu:
            self.model = model.cuda()
        else:
            self.model = model
        self.name = name # torch.typename(model)
        self.full_name = f'{name}_lr_{lr}__momentum_{momentum}__gamma_{gamma}__stepsize_{step_size}'.replace('.', '_')

        self.params = {
            'name': name,
            'lr': lr,
            'momentum': momentum,
            'gamma': gamma,
            'step_size': step_size,
        }
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)        

        print(f'use gpu: {use_gpu}')
        train_directory = os.path.join(root_directory, 'train')
        val_directory = os.path.join(root_directory, 'val')
        test_directory = os.path.join(root_directory, 'test')                
        log_directory = os.path.join(log_root_directory, self.full_name)
        self.model_result_file = os.path.join(log_directory, 'models')
        self.monitor = TensorboardMonitor(log_directory)

        train_dataset = datasets.ImageFolder(train_directory,
            transforms.Compose([
                transforms.Scale(256),
                transforms.RandomSizedCrop(224),
                # transforms.TenCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=no_workers)

        val_dataset = datasets.ImageFolder(val_directory,
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
            shuffle=True, num_workers=no_workers)


        self.test_loader = torch.utils.data.DataLoader(
            ImageFolder2(test_directory, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=no_workers, pin_memory=True)

    def train(self, num_epochs=50, early_stopping=10):
        print(f'training for {self.full_name}:')
        since = time.time()
        softmax = nn.Softmax()
        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        best_loss = 0.0
        best_epoch = 0
        data_loaders = {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
        }
        for epoch in range(num_epochs):
            self.monitor.epoch_start(epoch, num_epochs - 1)
            roc_loss = 0
            softmax = nn.Softmax()
            print(f'running epoch {epoch}...')
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for data in data_loaders[phase]:
                    # get the inputs
                    data_inputs, data_labels = data

                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(data_inputs.cuda())
                        labels = Variable(data_labels.cuda())
                    else:
                        inputs = Variable(data_inputs)
                        labels = Variable(data_labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # if phase == 'val':
                    #     predictions = softmax(outputs.data)[:, 1]
                    #     roc_loss = roc_loss + metrics.roc_auc_score(
                    #         data_labels.tolist(), predictions.data.tolist())

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

                dataset_size = len(data_loaders[phase])
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects / dataset_size

                if phase == 'val':
                    self.monitor.epoch_result(phase, epoch_loss, epoch_acc, roc_loss)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = self.model.state_dict()
                    self.save(epoch_acc, epoch_loss, epoch)

            # early stopping
            if epoch - best_epoch >= early_stopping:
                print('early stopping!')
                break        

                    
        time_elapsed = time.time() - since
        self.monitor.training_completed(time_elapsed, best_loss, best_acc)
        print(f'training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'best val acc: {best_acc:4f}, loss: {best_loss}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        result_file = os.path.join(self.model_result_file, f'{self.full_name}_model.csv')
        self.evaluate(None, result_file)

    def get_dict_file(self):
        return os.path.join(self.model_result_file, f'model.p')

    def save(self, epoch_acc: float, epoch_loss: float, epoch: int) -> None:
        if os.path.exists(self.model_result_file) == False:
            os.makedirs(self.model_result_file)

        dict_file = self.get_dict_file()
        info_file = os.path.join(self.model_result_file, f'result.json')
        
        if os.path.exists(dict_file):
            os.remove(dict_file)

        if os.path.exists(info_file):
            os.remove(info_file)
            
        torch.save(self.model.state_dict(), dict_file)
        with open(info_file, 'w') as file:
            params = dict(self.params)
            params['loss'] = epoch_loss,
            params['accurracy'] = epoch_acc,
            params['epoch'] = epoch
            file.write(json.dumps(params))

    def load(self, file_name: str) -> None:       
        self.model.load_state_dict(torch.load(file_name))

    def evaluate(self, model_file: str, output_file: str):
        if model_file != None:
            self.load(model_file)

        self.model.eval()
        softmax = nn.Softmax()
        list = []
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')            
            writer.writerow(['name', 'invasive'])
            for i, ((inputs, _), paths) in enumerate(self.test_loader):
                print(f'evaluating item {i}...')
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)

                # forward
                outputs = self.model(inputs)
                # _, predictions = torch.max(outputs.data, 1)
                # print(predictions)
                file_names = [os.path.splitext(os.path.basename(x))[0] for x in paths]
                predictions = softmax(outputs.data)[:, 1]
                print(f'evaluating item {i} batch_size: {len(file_names)}...')

                for file_name, prediction in zip(file_names, predictions.data.tolist()):                    
                    list.append([file_name, prediction])
                    writer.writerow([file_name, prediction])

        return list