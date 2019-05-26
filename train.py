from model_trainer import ModelTrainer
from logger import Logger

if __name__ == '__main__':

  logger = Logger(show = True, html_output = True, config_file = "config.txt",
    data_folder = "drive")

  trainer = ModelTrainer(logger)
  trainer.run_training(num_epochs = 50, save_after_epochs = 10)

