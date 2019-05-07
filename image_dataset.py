from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):

  def __init__(self, data_dict, logger, transform = None):

    self.logger = logger
    self.transform = transform
    self.X = data_dict['X']
    self.y = data_dict['y']


  def __len__(self):

    return len(self.y)


  def __getitem__(self, index):

    sample = {'image': self.X[index], 'label': self.y[index]}
    
    if self.transform:
      sample = self.transform(sample)

    return sample