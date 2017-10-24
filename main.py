# https://www.kaggle.com/c/invasive-species-monitoring

import os
from model import Model
from torchvision import models
import argparse
from tensorboard_monitor import TensorboardMonitor
data_root = '/workspace/data/species_monitoring'
logs = '/workspace/logs/species_monitoring'

options = [
    { 'lr': 0.001,   'momentum': 0.9,   'step_size': 5,    'gamma': 0.1  },
    # { 'lr': 0.001,   'momentum': 0.6,   'step_size': 5,    'gamma': 0.1  },
    # { 'lr': 0.001,   'momentum': 0.5,   'step_size': 5,    'gamma': 0.5   },
    # { 'lr': 0.01,    'momentum': 0.5,   'step_size': 5,    'gamma': 0.1  },
    # { 'lr': 0.1,     'momentum': 0.9,   'step_size': 5,    'gamma': 0.1  },
    # { 'lr': 0.1,     'momentum': 0.9,   'step_size': 5,    'gamma': 0.1  }
]

models_names = [
    # 'resnet18',
    # 'resnet34',
    # 'resnet50',
    # 'densenet121',

    'densenet169',
    'densenet201',
    'resnet101',
    'resnet152',
    'vgg16'
]

def get_model(model_name:str):
    return getattr(models, model_name)(pretrained=True)

def train_all(args):
    if os.path.exists(args.log_dir):
        os.rmdir(args.log_dir)

    for train_params in options:
        for model_name in models_names:
            model = get_model(model_name)
            model_instance = Model(
                data_root, 
                args.log_dir,
                model,
                model_name,
                train_params['lr'],
                train_params['momentum'],
                train_params['gamma'],
                train_params['step_size'])

            model_instance.train()

def evaluate(args):
    model_name = args.model_name
    model = get_model(model_name)
    model_instance = Model(
        data_root, 
        logs,
        model,
        model_name)
    model_instance.evaluate(args.model_file, args.output_file)

main_parser = argparse.ArgumentParser(prog='mynet', usage='%(prog)s [options]')
subparsers = main_parser.add_subparsers(help='commands')
train_parser = subparsers.add_parser('train', help='starts training')
train_parser.add_argument('log_dir', help='tensorboard log directory and results directory')
train_parser.set_defaults(func=train_all)
eval_parser = subparsers.add_parser('eval', help='starts evaluating')
eval_parser.set_defaults(func=evaluate)
eval_parser.add_argument('model_name', help='type of the model resnet151, ...')
eval_parser.add_argument('model_file', help='file where the model params are stored')
eval_parser.add_argument('output_file', help='output file')

args = main_parser.parse_args()
args.func(args)
