# Imports python modules
import argparse
from time import time, sleep
from os import listdir
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image 

def main():
    in_arg = get_input_args()
    loader = transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
   
    model = load_checkpoint(in_arg.checkpoint)
    
    probs, classes = predict(in_arg.input, model, in_arg.top_k, in_arg.gpu)
    
    cat_to_name = load_json(in_arg.category_names)

    
    print(probs)
    print(classes)
    if in_arg.gpu:
        y_pos = classes.data[0].cpu().numpy()
    else:
        y_pos = classes.data[0].numpy()
    
    flower_index = [model.class_to_idx[x] for x in y_pos]
    y_classes = [cat_to_name[y] for y in flower_index]

    print(y_classes)
# Functions defined below to get the input arguments
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 5 command line arguments args.input for input file to predict, 
    # checkpoint file to load the trained model, top_k to display number of top matches
    # category_names for json files containing flower name to lable mapping, GPU 
    parser.add_argument('input', metavar='input', type=str, 
                    help='path to input image file')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, 
                        help='checkpoint of previously trained model')
    parser.add_argument('--top_k', type=int, default=5, required=False,
                        help='top k predictions')
    parser.add_argument('--category_names', type=str, default='category_names cat_to_name.json',required=False,
                        help='Json file containing category names')
    parser.add_argument('--gpu', action='store_true', help='Run in GPU or not')
    
    # returns parsed argument collection
    return parser.parse_args()    
loader = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224),
                                     transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    from PIL import Image 
    if gpu:
        model.cuda()
    else:
        model.cpu()
    model.eval()
    img = process_image(image_path)
    image = Image.open(image_path)
    image = loader(image).float()
    image = Variable(image.unsqueeze(0))
    if gpu:
        output = model.forward(image.cuda())
    else:
        output = model.forward(image.cpu())
    ps = torch.exp(output) 
    return ps.topk(topk)[0],ps.topk(topk)[1]

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['model_arch']
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
#     print(model)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    mean = [0.485, 0.456, 0.406] 
    stdv = [0.229, 0.224, 0.225] 
    img = Image.open(image) 
  
    if img.size[0]>=img.size[1]: 
        img.thumbnail((10000,256)) 
    else: 
        img.thumbnail((256,10000))

    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img = img.crop(
        (
            half_the_width - 112,
            half_the_height - 112,
            half_the_width + 112,
            half_the_height + 112
        )
    )

    np_image = np.array(img)
    img = np_image/255
    img=(img-mean)/stdv

    img=img.transpose((2,0,1))
    return img

def load_json(category_file_name):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
# Call to main function to run the program
if __name__ == "__main__":
    main()