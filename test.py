from model_tester import ModelTester
from torchvision import transforms
from logger import Logger


if __name__ == '__main__':

  logger = Logger(show = True, html_output = True, config_file = "config.txt",
    data_folder = "drive")

  input_size = 32
  transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  tester = ModelTester(transform, logger)

  tester.run_tests()