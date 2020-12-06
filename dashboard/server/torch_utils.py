import json
import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Transform input into the form our model expects
def transform_image(infile):
    def resize(x):
        x = x.unsqueeze(0)
        x = ch.nn.functional.interpolate(x, size=args['resolution'], mode='bilinear')
        return x[0]

    my_transforms = transforms.Compose([           # We use multiple TorchVision transforms to ready the image
        transforms.ToTensor(),
        resize,
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])
    ])
    image = Image.open(infile).convert("RGB")                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg

# Get a prediction
def get_prediction(model, input_tensor):
    outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx, img_class_map):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name
