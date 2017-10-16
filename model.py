import os
import time
import csv
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from image_folder import ImageFolder2
from monitor import Monitor

batch_size = 10
no_workers = 4
use_gpu = torch.cuda.is_available()

class Model(object):
    def __init__(self, root_directory: str, model: nn.Module) -> None:                
        self.model = model #models.resnet152(pretrained=True)
        self.name = torch.typename(model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
        self.monitor = Monitor()

        model_result_directory = os.path.join(root_directory, 'results')
        train_directory = os.path.join(root_directory, 'train')
        val_directory = os.path.join(root_directory, 'val')
        test_directory = os.path.join(root_directory, 'test')
        time_part = time.strftime("%Y%m%d_%H%M%S")
        self.model_result_file = os.path.join(model_result_directory, f'result_{self.name}_{time_part}.torchdict')

        train_dataset = datasets.ImageFolder(train_directory,
            transforms.Compose([
                transforms.RandomSizedCrop(224),
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

    def train(self, num_epochs=25):
        since = time.time()

        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        data_loaders = {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
        }
        for epoch in range(num_epochs):
            self.monitor.epoch_start(epoch, num_epochs - 1)

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
                    inputs, labels = data

                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

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

                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

                dataset_size = len(data_loaders[phase])
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects / dataset_size

                self.monitor.epoch_result(phase, epoch_loss, epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
                    self.save_model()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model_result_file)

    def load_model(self, file_name: string = None) -> None:
        if file_name == None:
            file_name = self.model_result_file
        self.model.load_state_dict(torch.load(file_name))

    def evaluate_model(self):
        self.model.eval()
        softmax = nn.Softmax()
        list = []
        with open(csv_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['name', 'invasive'])
            for i, ((inputs, _), paths) in enumerate(test_loader):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)

                # forward
                outputs = self.model(inputs)
                # _, predictions = torch.max(outputs.data, 1)
                # print(predictions)
                file_names = [os.path.splitext(os.path.basename(x))[0] for x in paths[0]]
                predictions = softmax(outputs.data)[:, 1]
                
                for file_name, prediction in zip(file_names, predictions.data.tolist()):
                    list.append([file_name, prediction])
                    writer.writerow([file_name, prediction])

        return list