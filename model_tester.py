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
    self.results = {}


  def _load_device(self):
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def _load_trained_model(self):

    model_file = self.logger.config_dict['TRAINED_MODEL']
    self.logger.log("Start loading model from {}...".format(model_file))
    self.model = torch.load(self.logger.get_model_file(model_file))
    self.model.to(self.device)
    self.logger.log("Finish loading model", show_time = True)
    self.model_type = model_file.split('_')[0]


  def _load_labels_dict(self):

    df = pd.read_csv(self.logger.get_data_file(self.logger.config_dict['LABELS_FILE']))

    labels = df['ClassId'].values
    paths  = [os.path.basename(path) for path in df['Path'].values]

    self.labels_dict = dict(zip(paths, labels))


  def _load_orig_test_data(self):
    
    path_in_data_folder = self.logger.config_dict['TEST_FOLDER']
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


  def _get_mean_confidence(self):

    data_loader = torch.utils.data.DataLoader(self.crt_dataset, batch_size = 256, shuffle = False)
    self.model.eval()

    pred_mask, probs_list = [], []
    for batch in data_loader:
      inputs = batch['image']
      labels = batch['label']
      fnames = batch['fname']

      inputs_pt = inputs.to(self.device)
      labels_pt = labels.to(self.device)
      outputs_pt = self.model(inputs_pt)

      outputs_pt = torch.nn.Softmax(dim = 0)(outputs_pt)

      probs, preds = torch.max(outputs_pt, dim = 1)
      #probs, preds = probs.numpy(), preds.numpy()

      pred_mask  += (labels.cpu().numpy() == preds.cpu().numpy()).tolist()
      probs_list += probs.detach().cpu().numpy().tolist()


    wrong_img_idxs  = np.where(np.array(pred_mask) == False)[0].tolist()
    mask = np.ones(len(probs_list), dtype = bool)
    mask[wrong_img_idxs] = False
    correct_probs = np.array(probs_list)[mask]
    wrong_probs   = np.array(probs_list)[wrong_img_idxs]

    self.logger.log("Mean confidence for correct images: {:.2f}".format(
      correct_probs.mean()), tabs = 1)
    self.logger.log("Mean confidence for wrong images: {:.2f}".format(
      wrong_probs.mean()), tabs = 1)


  def _run_prediction(self):

    data_loader = torch.utils.data.DataLoader(self.crt_dataset, batch_size = 256, shuffle = False)
    self.model.eval()

    pred_mask, img_names, img_preds = [], [], []
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
      img_names  += fnames
      img_preds  += preds.cpu().numpy().tolist()


    wrong_img_idxs  = np.where(np.array(pred_mask) == False)[0].tolist()
    wrong_img_names = np.array(img_names)[wrong_img_idxs].tolist()

    crt_results = []
    for i, (img, pred) in enumerate(zip(img_names, img_preds)):
      if i in wrong_img_idxs:
        crt_results.append([img, self.labels_dict[img], str(pred)])
      if img not in self.results:
        self.results[img] = [self.labels_dict[img], str(pred)]
      else:
        self.results[img].append(str(pred))

    accuracy = sum(pred_mask) / self.crt_size

    self.logger.log("Accuracy at test: {:.2f}".format(accuracy), tabs = 1)
    self.logger.log("Number of wrong images: {}".format(len(wrong_img_idxs)), tabs = 1)
    filename_wrong_imgs = self.model_type + "_" + self.crt_folder +  "_wrong" + ".csv"
    self.logger.log("Save wrong images filenames to {}".format(filename_wrong_imgs))
    crt_results_df = pd.DataFrame(crt_results, columns = ["Name", "Orig_Label", "Pred_Label"])
    crt_results_df.to_csv(self.logger.get_output_file(filename_wrong_imgs), index = False)


  def _run_test_on_aug(self, folders_list):
  
    for folder in folders_list:
      self.logger.log("Test on {}".format(folder))
      self._load_aug_test_data(folder)
      self._get_mean_confidence()
      #self._run_prediction()


  def _run_test_on_orig(self):

    self.logger.log("Test on original")
    self._load_orig_test_data()
    self._get_mean_confidence()
    #self._run_prediction()


  def _save_results(self):

    rows = [np.concatenate(
      [[key], values]).tolist() for key, values in self.results.items()]
    self.rows = rows
    results_df = pd.DataFrame(rows, columns = self.res_columns)
    results_filename = self.model_type + "_results.csv"
    results_df.to_csv(self.logger.get_output_file(results_filename), index = False)


  def run_tests(self):

    self._run_test_on_orig()
    self.res_columns = ["Name", "Orig_Label", "Pred_Label"]

    blurred_folders = ["Blurred_" + str(i) for i in range(5, 25, 5)]
    self._run_test_on_aug(blurred_folders)
    self.res_columns += ["Pred_" + folder + "_Label" for folder in blurred_folders]

    dark_folders = ["Dark_" + str(i) for i in range(1, 4)]
    self._run_test_on_aug(dark_folders)
    self.res_columns += ["Pred_" + folder + "_Label" for folder in dark_folders]

    bright_folders = ["Bright_" + str(i) for i in range(1, 8)]
    self._run_test_on_aug(bright_folders)
    self.res_columns += ["Pred_" + folder + "_Label" for folder in bright_folders]

    occlud_folders = ["Occl_" + str(i) for i in range(5, 30, 5)]
    self._run_test_on_aug(occlud_folders)
    self.res_columns += ["Pred_" + folder + "_Label" for folder in occlud_folders]

    #self._save_results()