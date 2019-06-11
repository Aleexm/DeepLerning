import torch
from torch.nn import ReLU

from layervis_utils import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

from logger import Logger

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import imageio
import os


class GuidedBackprop():
  """
    Produces gradients generated with guided back propagation from the given image
  """
  def __init__(self, model):

    self.model = model
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.gradients = None
    self.forward_relu_outputs = []
    # Put model in evaluation mode
    self.model.eval()
    self.update_relus()
    self.hook_layers()

  def hook_layers(self):
    def hook_function(module, grad_in, grad_out):
      self.gradients = grad_in[0]
    # Register hook to the first layer
    first_layer = list(self.model.features._modules.items())[0][1]
    first_layer.register_backward_hook(hook_function)

  def update_relus(self):
    """
      Updates relu activation functions so that
        1- stores output in forward pass
        2- imputes zero for gradient values that are less than zero
    """
    def relu_backward_hook_function(module, grad_in, grad_out):
      """
        If there is a negative gradient, change it to zero
      """
      # Get last forward output
      corresponding_forward_output = self.forward_relu_outputs[-1]
      corresponding_forward_output[corresponding_forward_output > 0] = 1
      modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
      del self.forward_relu_outputs[-1]  # Remove last forward output
      return (modified_grad_out,)

    def relu_forward_hook_function(module, ten_in, ten_out):
      """
        Store results of forward pass
      """
      self.forward_relu_outputs.append(ten_out)

    # Loop through layers, hook up ReLUs
    for pos, module in self.model.features._modules.items():
      if isinstance(module, ReLU):
        module.register_backward_hook(relu_backward_hook_function)
        module.register_forward_hook(relu_forward_hook_function)

  def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
    self.model.zero_grad()
    # Forward pass
    x = input_image
    x = x.to(self.device)
    for index, layer in enumerate(self.model.features):
      # Forward pass layer by layer
      # x is not used after this point because it is only needed to trigger
      # the forward hook function
      x = layer(x)
      # Only need to forward until the selected layer is reached
      if index == cnn_layer:
        # (forward hook function triggered)
        break
    conv_output = torch.sum(torch.abs(x[0, filter_pos]))
    # Backward pass
    conv_output.backward()
    # Convert Pytorch variable to numpy array
    # [0] to get rid of the first channel (1,3,224,224)
    gradients_as_arr = self.gradients.data.cpu().numpy()[0]
    return gradients_as_arr


if __name__ == '__main__':
  
  logger = Logger(show = True, html_output = True, config_file = "config.txt",
    data_folder = "drive")

  filter_pos = 1

  model_file = logger.config_dict['TRAINED_MODEL']
  logger.log("Start loading model from {}...".format(model_file))
  pretrained_model = torch.load(logger.get_model_file(model_file))
  logger.log("Finish loading model", show_time = True)
  
  #[("Blur-VGG", 4) , ("E-VGG", 8) , ("G-VGG", 12)]
  folder, num_images = ("E-VGG_new", 2)

  for i in range(num_images):
    (original_image, prep_img, target_class, base_file_name_to_export) =\
      get_example_params(folder, i, logger)
    resulted_images = []
    titles = []
    for layer_idx in range(len(pretrained_model.features)):
      file_name_to_export = base_file_name_to_export + folder.lower() + "_" +  '_layer' + str(layer_idx) + '_filter' + str(filter_pos)
      GBP = GuidedBackprop(pretrained_model)
      # Get gradients
      guided_grads = GBP.generate_gradients(prep_img, target_class, layer_idx, filter_pos)
      # Save colored gradients
      save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
        
      resulted_images.append(file_name_to_export + '_Guided_BP_color.jpg')
      titles.append("Layer{}".format(layer_idx))
      
    logger.log('Layer visualization for image {} from {} completed'.format(i, folder))

    for filename, title in zip(resulted_images, titles):
      img = Image.open(filename)
      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype("arial.ttf", 10)
      draw.text((0, 0), title,(0, 0, 255),font=font)
      img.save(filename)

    images = []
    for filename in resulted_images:
      images.append(imageio.imread(filename))

    filename = folder.lower() + "_" + os.path.basename(base_file_name_to_export) + ".gif"
    filename = logger.get_output_file(filename)
    imageio.mimsave(filename, images, duration = 0.5)

    torch.cuda.empty_cache()