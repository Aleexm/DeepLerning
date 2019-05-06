import torch.nn as nn
from torchvision.models import vgg16

def convolutionize(model, num_classes, input_size=(3, 224, 224)):
    '''Converts the classification layers of VGG & Alexnet to convolutions

    Input:
        model: torch.models
        num_classes: number of output classes
        input_size: size of input tensor to the model

    Returns:
        model: converted model with convolutions
    '''
    features = model.features
    classifier = model.classifier

    # create a dummy input tensor and add a dim for batch-size
    x = torch.zeros(input_size).unsqueeze_(dim=0)

    # change the last layer output to the num_classes
    classifier[-1] = nn.Linear(in_features=classifier[-1].in_features,
                               out_features=num_classes)

    # pass the dummy input tensor through the features layer to compute the output size
    for layer in features:
        x = layer(x)

    conv_classifier = []
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            # create a convolution equivalent of linear layer
            conv_layer = nn.Conv2d(in_channels=x.size(1),
                                   out_channels=layer.weight.size(0),
                                   kernel_size=(x.size(2), x.size(3)))

            # transfer the weights
            conv_layer.weight.data.view(-1).copy_(layer.weight.data.view(-1))
            conv_layer.bias.data.view(-1).copy_(layer.bias.data.view(-1))
            layer = conv_layer

        x = layer(x)
        conv_classifier.append(layer)

    # replace the model.classifier with newly created convolution layers
    model.classifier = nn.Sequential(*conv_classifier)

    return model



model = vgg16()

new_model = con


