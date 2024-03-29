# Deep learning project "Strike with BLOD" :vertical_traffic_light:

Code for Group 16 python implementation of "Strike with BLOD: Analyzing the impact of **B**lur, **L**ightness, **O**cclusion, and **D**arkness on traffic sign recognition performance of CNN networks" as part of CS4180 Deep Learning 2018-2019 :mortar_board:.

Team members:

 * [Andrei Simion-Constantinescu](https://www.linkedin.com/in/andrei-simion-constantinescu/)
 * [Nele Albers](https://github.com/nelealbers)
 * [Alex Mandersloot](https://github.com/Aleexm)
 * [Stefan Bonhof](https://github.com/SDBonhof)
 * [Joram van der Sluis](https://github.com/joramvdsluis)
 
 ## Data :floppy_disk:
 
 Due to the large size of our datasets, we did not upload the data on GitHub. The project structure enforces the data to be in a local Google Drive folder. The folder can be placed in the following accepted locations: `home_dir/Google Drive`, `home_dir/GoogleDrive`, `home_dir/Desktop/Google Drive`, `home_dir/Desktop/GoogleDrive`, `C:/Google Drive`, `C:/GoogleDrive`, `D:/Google Drive`, `D:/GoogleDrive`, where **home_dir** is the path to the user home directory. Inside the Google Drive folder, the application folder needs to be created, `Deep-Learning-Team16`. The data archieve can be downloaded from this [link](https://drive.google.com/file/d/18ZK4E9jfKA8pvgsDfyMq0CiOxbw7Iq9B/view?usp=sharing). Unzip the downloaded archieve and place it in the newly created application folder, `google_drive_path/Deep-Learning-Team16/`.
 
 ## Project structure :open_file_folder:
 
 The structure of the project is presented per tasks:
 
 ### Training
 
 * `model_trainer.py` - implementation of the training pipeline class
 * `train.py` - main file for running a training session with the parameters read from configuration file
 * `image_dataset.py` - pytorch Dataset class implementation for loading and processing our images

### Testing

* `model_tester.py` - implementation of the testing pipeline class
* `test.py` - main file for running a testing session with the parameters read from configuration file
* `plot_utils.py` - script for generating [Bokeh](https://bokeh.pydata.org/en/latest/) confusion matrices
* `layervis_utils.py` - utils functions for feature map visualization inspired by [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* `layer_visualization.py` - main file for creating feature maps visusalization and saving them to `layervis_results` folder

### Analysis

* `analysisPlotFunctions.py` - implementation of all analysis plot functions
* `analysis.py` - main file for calling analysis plot functions to plot class accuracy, predictions a.s.o

### Data augmentation

* `Image-augmentation.py` - contains functions to load images, augment them with BLOD, and saves them seperately.

### Others
* `logger.py` -  logging system for generating folders initial structure and saving application logs to HTML files
* `config.txt` - configuration file

### Additional folders
* `layervis_results\` - for storing feature maps figures
* `output\` - for storing results for each tested arhitecture with different training and testing data
> :exclamation: For each tested scenario, we have log file from training, testing, filename list of wrongly classified images, confusion matrices in Bokeh, aggregated results, training and validation loss per epoch
* `models\` - saved GPU trained models for each scenario
> :exclamation: Not included on GitHub due to space limitation
 
## Config file :bookmark_tabs:
 
 ```
{
  "APP_FOLDER" : "Deep-Learning-Team16",
  "DATA_FOLDER"  : "data",
  "MODEL_TO_TRAIN": "VGG16",
  "NUM_CLASSES" : "43",
  "INPUT_SIZE": 32,
  "TRAIN_FOLDER": "Train",
  "ADDITIONAL_TRAIN_FOLDER": "European",
  "TEST_FOLDER": "Test",
  "LABELS_FILE": "Test.csv",
  "AUG_TEST_FOLDER": "Augmented_test_sets",
  "TRAINED_MODEL": "VGG16ADDEUR_BEST_E48_L0.05_2019-05-20_19_38_15.ptm",
  "RESULTS_FILE": "VGG16ADDEUR_results.csv",
  "TRAIN_TYPE": "ADDEUR",
  "IMAGES_FEAT_MAPS": "Images_feature_maps"
}
```

## Installation :computer:

The scripts can be run in [Anaconda](https://www.anaconda.com/download/) Windows/Linux environment.

You need to create an Anaconda :snake: `python 3.6` environment named `dl_gts`.
Inside that environment some addition packages needs to be installed. Run the following commands inside Anaconda Prompt ⌨:
```shell
(base) conda create -n dl_gts python=3.6 anaconda
(base) conda activate dl_gts
(dl_gts) conda install -c pytorch pytorch
(dl_gts) conda install -c pytorch torchvision
(dl_gts) conda install -c anaconda cudatoolkit
(dl_gts) conda install -c conda-forge tqdm 
(dl_gts) conda install -c conda-forge opencv
```

For GPU support, NVIDIA CUDA compatible graphic card is needed with proper drivers installed.

## Usage :arrow_forward:

For the training pipeline, the following configuration file fields can be modified:
* "MODEL_TO_TRAIN": VGG16 - pretrained on ImageNet, CUSTOMVGG - pretrained VGG16 without fully-connected layers, DENSENET - pretrained on ImageNet
* "TRAIN_TYPE": INITIAL_TRAINING - train on german training dataset, ADDEUR/ADDBLUR/ADDDARK, ADDLIGHT, ADDOCCL - train on german training dataset + additional data either european or european BLOD augmented, ADDALL - train on german training dataset + additional all randomly chosen BLOD data
* "ADDITIONAL_TRAIN_FOLDER": European/European_blurred/European_dark, European_light, European_occluded to be modifed when "TRAIN_TYPE" is ADDEUR/ADDBLUR/ADDDARK, ADDLIGHT, ADDOCCL

After setting the desired scenario, simply run `python train.py` from `dl_gts` enviroment.

For the testing pipeline, the following configuration file fields can be modified:
* "TRAINED_MODEL": the saved trained model from `models/` folder for which the results are generated
* "RESULTS_FILE": the aggregated results file generated after running `python test.py` used for creating the confusion matrices

After setting the desired scenario, simply run from `dl_gts` enviroment `python test.py` for running predictions on all testing datasets and `python plot_utils.py` for generating Bokeh confusion matrices after the aggregated results file was created from running the predictions.



