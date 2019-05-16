import pandas as pd
import numpy as np
import torch
import cv2
import csv
import os

from image_dataset import ImageDataset


class ModelTester():

  def __init__(self, data_transform, logger):

    self.model = None
    self.data_transform = data_transform
    self.logger = logger

    self._load_device()
    self._load_trained_model()
    self._load_labels_dict()


  def _load_device(self):
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def _load_trained_model(self):

    self.logger.log("Start loading model...")
    model_file = self.logger.config_dict['TRAINED_MODEL']
    self.model = torch.load(self.logger.get_model_file(model_file))
    self.model.to(self.device)
    self.logger.log("Finish loading model", show_time = True)


  def _load_labels_dict(self):

    df = pd.read_csv(self.logger.get_data_file(self.logger.config_dict['LABELS_FILE']))

    labels = df['ClassId'].values
    paths  = [os.path.basename(path) for path in df['Path'].values]

    self.labels_dict = dict(zip(paths, labels))


  def _load_orig_test_data(self):
    
    path_in_data_folder = "Test"
    self._load_test_data(path_in_data_folder)


  def  _load_aug_test_data(self, folder_name):
    path_in_data_folder = os.path.join(self.logger.config_dict['AUG_TEST_FOLDER'], 
      folder_name)
    self._load_test_data(path_in_data_folder)


  def _load_test_data(self, path_in_data_folder):

    data, labels, fnames = [], [], []

    full_path = os.path.join(self.logger.data_folder, path_in_data_folder)
    enc_path = os.fsencode(full_path)
    for file in os.listdir(enc_path):
      filename = os.fsdecode(file)
      if filename.endswith(".png"):
        data.append(cv2.imread(self.logger.get_data_file(filename, path_in_data_folder)))
        labels.append(self.labels_dict[filename])
        fnames.append(filename)

    data_dict = {'X': data, 'y': labels}
    self.crt_dataset = ImageDataset(data_dict, self.logger, transform = self.data_transform, 
      fnames = fnames)
    self.crt_folder = os.path.basename(path_in_data_folder)
    self.crt_size = len(fnames)


  def _run_prediction(self):

    data_loader = torch.utils.data.DataLoader(self.crt_dataset, batch_size = 256, shuffle = False)

    pred_mask, wrong_idx, wrong_imgs = [], [], []
    for batch in data_loader:
      inputs = batch['image']
      labels = batch['label']
      fnames = batch['fname']

      inputs_pt = inputs.to(self.device)
      labels_pt = labels.to(self.device)
      outputs_pt = self.model(inputs_pt)

      probs, preds = torch.max(outputs_pt, dim = 1)
      #probs, preds = probs.numpy(), preds.numpy()

      pred_mask  += (labels.cpu().numpy() == preds.cpu().numpy()).tolist()
      wrong_imgs += fnames

    wrong_idx = np.where(np.array(pred_mask) == False)[0].tolist()
    wrong_imgs = np.array(wrong_imgs)[wrong_idx].tolist()
    wrong_imgs = [e.encode() for e in wrong_imgs]
    accuracy = sum(pred_mask) / self.crt_size

    self.logger.log("Accuracy at test: {:.2f}".format(accuracy), tabs = 1)
    self.logger.log("Number of wrong images: {}".format(len(wrong_imgs)), tabs = 1)
    filename_wrong_imgs = self.crt_folder +  "_wrong" + ".csv"
    self.logger.log("Save wrong images filenames to {}".format(filename_wrong_imgs))
    with open(self.logger.get_output_file(filename_wrong_imgs), 'a', newline = '') as fp:
      writer = csv.writer(fp, delimiter=',', dialect='excel')
      for row in wrong_imgs:
        writer.writerow([str(row)])
        #writer.writerow(os.linesep)


  def _run_test_on_aug(self, folders_list):
  
    for folder in folders_list:
      self.logger.log("Test on {}".format(folder))
      self._load_aug_test_data(folder)
      self._run_prediction()


  def _run_test_on_orig(self):

    self.logger.log("Test on original")
    self._load_orig_test_data()
    self._run_prediction()


  def run_tests(self):

    self._run_test_on_orig()

    blurred_folders = ["Blurred_" + str(i) for i in range(5, 20, 5)]
    self._run_test_on_aug(blurred_folders)

    bright_folders = ["Bright_" + str(i) for i in range(1, 11)]
    self._run_test_on_aug(bright_folders)

    occlud_folders = ["Occl_" + str(i) for i in range(5, 30, 5)]
    self._run_test_on_aug(occlud_folders)