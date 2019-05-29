import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, trange
import math
import numpy as np
import pickle as pkl
import inspect
import os
import cv2
import copy
import time

from sklearn.model_selection import train_test_split

import torch
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

from image_dataset import ImageDataset
from logger import Logger


class ModelTrainer():

  def __init__(self, logger):

    self.logger = logger
    self.train_type = self.logger.config_dict['TRAIN_TYPE']
    self.data, self.labels = None, None
    self._load_model()
    self._load_device()
    if self.train_type == "INITIAL_TRAINING":
      self._load_data()
    else:
      self._load_additional_data()
      self._load_data()
      self.model_type += self.train_type


  def _load_device(self):
  
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.logger.log("Set device to {}".format(self.device))
  

  def _load_model(self):

    self.model_type = self.logger.config_dict['MODEL_TO_TRAIN']
    self.input_size = int(self.logger.config_dict['INPUT_SIZE'])
    self.num_classes = int(self.logger.config_dict['NUM_CLASSES'])
    if self.model_type == "VGG16":
      self._load_vgg()
    elif self.model_type == "CUSTOMVGG":
      self._load_custom_vgg()
    elif self.model_type == "DENSENET":
      self._load_densenet()


  def _load_custom_vgg(self, pretrained = True):

    self.logger.log("Start loading {} custom VGG model ...".format(
      "pretrained" if pretrained else ""))
    self.model = models.vgg16(pretrained = pretrained)
    self._set_parameter_requires_grad(feature_extracting = False)
    self.model.classifier = nn.Linear(self.model.classifier[0].in_features, self.num_classes)
    self.logger.log("Finished loading model", show_time = True)

    self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(self.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    self.valid_transform = transforms.Compose([
        transforms.Resize(self.input_size),
        transforms.CenterCrop(self.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


  def _load_vgg(self, pretrained = True):

    self.logger.log("Start loading {} VGG16 model ...".format(
      "pretrained" if pretrained else ""))
    self.model = models.vgg16(pretrained = pretrained)
    self._set_parameter_requires_grad(feature_extracting = False)
    num_feats = self.model.classifier[6].in_features
    self.model.classifier[6] = nn.Linear(num_feats, self.num_classes)
    self.logger.log("Finished loading model", show_time = True)
    

    self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(self.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    self.valid_transform = transforms.Compose([
        transforms.Resize(self.input_size),
        transforms.CenterCrop(self.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


  def _load_densenet(self, pretrained = True):

    self.logger.log("Start loading {} DenseNet model ...".format(
      "pretrained" if pretrained else ""))
    self.model = models.densenet121(pretrained = pretrained)
    self._set_parameter_requires_grad(feature_extracting = False)
    num_feats = self.model.classifier.in_features
    self.model.classifier = nn.Linear(num_feats, self.num_classes)
    self.logger.log("Finished loading model", show_time = True)
    
    self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(self.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    self.valid_transform = transforms.Compose([
        transforms.Resize(self.input_size),
        transforms.CenterCrop(self.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



  def _split_train_valid(self, batch_size = 256):

    self.logger.log("Splitting in 90%/10% train/validation")
    X_train, X_valid, y_train, y_valid = train_test_split(
      self.data, self.labels, test_size = 0.10, random_state = 13)

    train_dict = {'X': X_train, 'y': y_train}
    valid_dict = {'X': X_valid, 'y': y_valid}

    train_dataset = ImageDataset(train_dict, self.logger,
      transform = self.train_transform)
    valid_dataset = ImageDataset(valid_dict, self.logger,
      transform = self.valid_transform)

    self.dataloaders = {
      'train': 
        torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True),
      'valid':
        torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    }
    self.data, self.labels = None, None


  def _load_additional_data(self):

    data, labels = [], []
    train_folder = self.logger.config_dict['ADDITIONAL_TRAIN_FOLDER']

    self.logger.log("Start loading additional data from {}...".format(
      train_folder))
    directory = os.path.join(self.logger.data_folder, train_folder)

    for subdir in os.listdir(directory):
      if os.path.isdir(os.path.join(directory, subdir)):
        crt_label = int(subdir.split('-')[-1])
        for file in os.listdir(os.path.join(directory, subdir)):
          if file.endswith(".ppm") and file[0] != 'G':
            data.append(cv2.imread(self.logger.get_data_file(
              file, os.path.join(train_folder, subdir))))
            labels.append(crt_label)

    self.logger.log("Finished loading data: {} entries".format(len(labels)), show_time = True)
    self.data, self.labels = data, labels


  def _load_data(self):

    data, labels = [], []
    train_folder = self.logger.config_dict['TRAIN_FOLDER']

    self.logger.log("Start loading data from {}...".format(train_folder))
    for i in tqdm(range(self.num_classes)):
      directory = os.fsencode(os.path.join(
        self.logger.data_folder, train_folder, str(i)))

      for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
          data.append(cv2.imread(self.logger.get_data_file(filename, os.path.join(
            train_folder, str(i)))))
          labels.append(i)

    self.logger.log("Finished loading data: {} entries".format(len(labels)), show_time = True)
    if self.data is not None:
      self.data += data
      self.labels += labels
    else: 
      self.data = data 
      self.labels = labels
    
    self._split_train_valid()


  def _set_parameter_requires_grad(self, feature_extracting):
  
    if feature_extracting:
      for param in self.model.parameters():
        param.requires_grad = False


  def _train_model(self, num_epochs, save_after_epochs = 1):
    
    old_print = print
    inspect.builtins.print = tqdm.write
    t = trange(num_epochs, desc='Epoch bar', leave=True)

    validation_acc_list, validation_loss_list, training_loss_list  = [], [], []
    best_acc = 0
    for epoch in range(num_epochs):
      start_time = time.time()
      t.set_description("Epoch #{}/{}".format(epoch + 1, num_epochs))
      t.refresh()
      t.update(1)

      inspect.builtins.print = old_print

      self.logger.log("Start epoch #{}".format(epoch + 1))

      for phase in ['train', 'valid']:

        if phase == "train":
          self.model.train()
        else:
          self.model.eval()

        epoch_loss = 0
        epoch_corrects = 0
        
        for dataset in self.dataloaders[phase]: 
          inputs = dataset['image']
          labels = dataset['label']
          inputs_pt = inputs.to(self.device)
          labels_pt = labels.to(self.device)

          self.optimizer.zero_grad()

          with torch.set_grad_enabled(phase == "train"):
            outputs_pt = self.model(inputs_pt)
            loss_pt = self.criterion(outputs_pt, labels_pt)

            _, preds = torch.max(outputs_pt, dim = 1)

            if phase == "train":
              loss_pt.backward()
              self.optimizer.step()

          epoch_loss      += loss_pt.item() * inputs_pt.size(0)
          epoch_corrects  += torch.sum(preds == labels_pt.data)

        epoch_loss /= len(self.dataloaders[phase].dataset) 
        epoch_acc  = epoch_corrects.double() / len(self.dataloaders[phase].dataset)

        self.logger.log("{} loss at epoch {}: {}".format(
          "Train" if phase == "train" else "Validation", epoch + 1, epoch_loss))
        self.logger.log("{} accuracy at epoch {}: {}".format(
          "Train" if phase == "train" else "Validation", epoch + 1, epoch_acc))

        if phase == "valid" and epoch_acc > best_acc:
          best_acc   = epoch_acc
          best_epoch = epoch
          best_loss  = epoch_loss
          best_model = copy.deepcopy(self.model)

        if phase == "valid":
          if epoch % save_after_epochs == 0:
            self.logger.log("Save model at epoch {}".format(epoch + 1))
            self.logger.save_model(self.model, self.model_type, epoch + 1, epoch_loss)

          validation_acc_list.append(epoch_acc) 
          validation_loss_list.append(epoch_loss)
        else:
          training_loss_list.append(epoch_loss)

      self.logger.log("Finish epoch no {} in {:.2f}s".format(epoch + 1, time.time() - start_time))

    t.close()
    self.logger.log("Save best model")
    self.logger.save_model(best_model, self.model_type + "_BEST", best_epoch + 1, best_loss)
    self.model = copy.deepcopy(best_model)

    self._plot_loss_history(training_loss_list, validation_loss_list)
    

  def _plot_loss_history(self, training_list, validation_list):

    fig = plt.figure(figsize=(5, 5))
    sns.set()

    tmstp = self.logger.get_time_prefix()
    with open(self.logger.get_output_file(self.model_type + "_train_loss_" + tmstp + ".pkl"), 'wb') as fp:
      pkl.dump(training_list, fp)
    with open(self.logger.get_output_file(self.model_type + "_valid_loss_" + tmstp + ".pkl"), 'wb') as fp:
      pkl.dump(validation_list, fp)

    epochs = np.array(range(len(training_list)))
    if len(training_list) <= 20:
      X = np.array(range(len(training_list)))
    else:
      factor = math.ceil(len(training_list) / 20) 
      X = np.array([i for i in range(0, len(training_list), factor)])

    plt.title("Loss during training of {}".format(self.model_type))
    plt.plot(epochs, training_list, linestyle='--', color='blue', label = 'Training')
    plt.plot(epochs, validation_list, linestyle='-', color='red', label = 'Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(X, (X + 1).tolist()) 

    plt.legend(loc = 'upper right')

    filename = self.model_type + "_train_valid_loss_" + tmstp + ".jpg"
    plt.savefig(self.logger.get_output_file(filename), dpi = 120, 
      bbox_inches='tight')


  def run_training(self, num_epochs, save_after_epochs):

    self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    self.criterion = nn.CrossEntropyLoss()
    self._train_model(num_epochs, save_after_epochs)