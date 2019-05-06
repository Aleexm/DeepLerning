from tqdm import tqdm
import os
import cv2

from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):

	def __init__(self, logger, transform = None):

		self.logger = logger
		self.transform = transform
		self.data, self.labels = self._read_data()


	def _read_data(self):

		data = []
		labels = []
  	for i in tqdm(range(int(self.logger.config_dict['NUM_CLASSES']))):
   		directory = os.fsencode(os.path.join(self.logger.data_folder, "Train", str(i)))

   		for file in os.listdir(train_directory):
      	filename = os.fsdecode(file)
      	if filename.endswith(".png"):
        	data.append(cv2.imread(self.logger.get_data_file(filename, os.path.join(
          	"Train", str(i)))))
        	labels.append(i)

    return data, labels


  def __len__(self):

  	return len(self.data)


  def __getitem__(self, index):

  	sample = {'image': self.data[index], 'label': self.data[index]}
   	
   	if self.transform:
   		sample = self.transform(sample)

   	return sample