# Deep learning project "Srike with BLOD" :vertical_traffic_light:

Code for Group 16 python implementation of "Strike with BLOD: Analyzing the impact of Blur, Lightness, Occlusion, and Darkness on traffic sign recognition performance of CNN networks" as part of CS4180 Deep Learning 2018-2019 :mortar_board:.

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

### Testing

* `model_tester.py` - implementation of the testing pipeline class
* `test.py` - main file for running a testing session with the parameters read from configuration file
* `plot_utils.py` - script for generating [Bokeh](https://bokeh.pydata.org/en/latest/) confusion matrices
* `layervis_utils.py` - utils functions for feature map visualization inspired by [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* `layer_visualization.py` - main file for creating feature maps visusalization and saving them to `layervis_results` folder

 
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

## Usage :arrow_forward:
