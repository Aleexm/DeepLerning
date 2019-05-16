from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):

  def __init__(self, data_dict, logger, transform = None, fnames = None):

    self.logger = logger
    self.transform = transform
    self.X = data_dict['X']
    self.y = data_dict['y']
    self.fnames = fnames


  def __len__(self):

    return len(self.y)


  def __getitem__(self, index):
      
    sample = self.X[index]
    
    if self.transform:
        sample = np.array(sample) # convert to numpy array
        sample = Image.fromarray(sample) # convert to PIL image
        sample = self.transform(sample)
     
    if self.fnames is None: 
        sample = {'image': sample, 'label': self.y[index]}
    else:
        sample = {'image': sample, 'label': self.y[index], 'fname': self.fnames[index]}

    return sample