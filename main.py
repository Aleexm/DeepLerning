from utils import get_drive_path
from logger import Logger

from tqdm import tqdm, trange
import inspect

import torchvision.models as models
from torchvision import transforms
import torch.optim as optim


import torch.nn as nn
import torch

from model_utils import set_parameter_requires_grad

import os
import cv2
import copy


from sklearn.model_selection import train_test_split
from image_dataset import ImageDataset


def read_data(logger):

  data = []
  labels = []
  for i in tqdm(range(int(logger.config_dict['NUM_CLASSES']))):
    directory = os.fsencode(os.path.join(logger.data_folder, "Train", str(i)))

    for file in os.listdir(directory):
      filename = os.fsdecode(file)
      if filename.endswith(".png"):
        data.append(cv2.imread(logger.get_data_file(filename, os.path.join(
          "Train", str(i)))))
        labels.append(i)

  return data, labels


def train_model(logger, model, dataloaders, criterion, optimizer, num_epochs):

  old_print = print
  inspect.builtins.print = tqdm.write
  t = trange(num_epochs, desc='Epoch bar', leave=True)

  validation_acc_list = []
  best_acc = 0
  for epoch in range(num_epochs):
    t.set_description("Epoch no {}/{}".format(epoch, num_epochs))
    t.refresh()
    t.update(1)

    inspect.builtins.print = old_print

    logger.log("Start epoch no {}".format(epoch))

    for phase in ['train', 'valid']:

      if phase == "train":
        model.train()
      else:
        model.eval()

      epoch_loss = 0
      epoch_corrects = 0
      
      for dataset in dataloaders[phase]: # fixed here
        inputs = dataset["image"]
        labels = dataset["label"]
        inputs_pt = inputs.to(device)
        labels_pt = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs_pt = model(inputs_pt)
          loss_pt = criterion(outputs_pt, labels_pt)

          _, preds = torch.max(outputs_pt, dim = 1)

          if phase == "train":
            loss_pt.backward()
            optimizer.step()

        epoch_loss      += loss_pt.item() * inputs_pt.size(0)
        epoch_corrects  += torch.sum(preds == labels_pt.data)

      epoch_loss /= len(dataloaders[phase].dataset) 
      epoch_acc  = epoch_corrects.double() / len(dataloaders[phase].dataset)

      logger.log("{} loss at epoch {}: {}".format(
        "Train" if phase == "train" else "Validation", epoch, epoch_loss))
      logger.log("{} accuracy at epoch {}: {}".format(
        "Train" if phase == "train" else "Validation", epoch, epoch_acc))

      if phase == "valid" and epoch_acc > best_acc:
        best_acc   = epoch_acc
        best_model = copy.deepcopy(model.state_dict())

      if phase == "valid":
        validation_acc_list.append(epoch_acc) 

    logger.log("Finish epoch no {}".format(epoch), show_time = True)

  return model.load_state_dict(best_model), validation_acc_list


def load_vgg(logger):

  model = models.vgg16(pretrained=True)
  set_parameter_requires_grad(model, feature_extracting = False)
  num_feats = model.classifier[6].in_features
  model.classifier[6] = nn.Linear(num_feats, int(logger.config_dict['NUM_CLASSES']))

  return model


if __name__ == '__main__':

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  data_folder = os.path.join(get_drive_path(), logger.config_dict['APP_FOLDER'], 
    logger.config_dict['DATA_FOLDER'])
  print(data_folder)
  logger.data_folder = data_folder
  
  num_classes = int(logger.config_dict['NUM_CLASSES'])
  model = load_vgg(logger)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  data, labels = read_data(logger)


  X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size = 0.1,
    random_state = 13)

  train_dict = {'X': X_train, 'y': y_train}
  valid_dict = {'X': X_valid, 'y': y_valid}

  input_size = 32
  
  train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  train_dataset = ImageDataset(train_dict, logger, transform = train_transform)
  valid_dataset = ImageDataset(valid_dict, logger, transform = valid_transform)

    # took num_workers out here
  dataloaders = {
    'train': 
      torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True),
    'valid':
      torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle = True)
    }

  for name,param in model.named_parameters():
    if param.requires_grad == True:
      logger.log("Param to optimize {}".format(name))

  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  criterion = nn.CrossEntropyLoss()

  trained_model, hist = train_model(logger, model, dataloaders, criterion, optimizer, num_epochs = 1)