from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):

  def __init__(self, data_dict, logger, transform = None):

    self.logger = logger
    self.transform = transform
    self.X = data_dict['X']
    self.y = data_dict['y']


  def __len__(self):

    return len(self.y)


  def __getitem__(self, index):
      
    sample = self.X[index]
    
    if self.transform:
        sample = np.array(sample) # convert to numpy array
        sample = Image.fromarray(sample) # convert to PIL image
        sample = self.transform(sample)
      
    sample = {'image': sample, 'label': self.y[index]}

    return sample