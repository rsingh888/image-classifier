# Imports python modules
import argparse
from time import time, sleep
from os import listdir

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

def main():
    in_arg = get_input_args()
#     print("data directory {}".format(in_arg.data_dir))
#     print(in_arg.epochs)
#     print(in_arg.hidden_units)
#     print(in_arg.arch)
#     print(in_arg.gpu)
    
    train_n_save_model(in_arg.arch, in_arg.learning_rate, 
                       in_arg.hidden_units, in_arg.epochs, 
                       in_arg.gpu, in_arg.data_dir,  in_arg.save_dir)
    
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

    # Creates 7 command line arguments args.dir for path to images files,
    # path to save trained model, 
    # args.arch which CNN model to use for classification, learning_rate,
    # number of hidden units, epochs for training, GPU 
    parser.add_argument('data_dir', metavar='data_dir', type=str, 
                    help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='assets/', required=False,
                        help='path to save trained model')
    parser.add_argument('--arch', type=str, default='densenet121', required=False,
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,required=False,
                        help='Learning rate to train the model')
    parser.add_argument('--hidden_units', type=int, default=400,required=False,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3,required=False,
                        help='Number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Run in GPU or not')
    
    # returns parsed argument collection
    return parser.parse_args()    

def train_n_save_model(arch, learning_rate, hidden_units, epochs, gpu, data_dir, save_dir):
    model = eval("models." + arch + "(pretrained=True)")
#     print(model)
    for param in model.parameters():
        param.requires_grad = False

    if isinstance(model.classifier, nn.modules.container.Sequential):
        input_units = model.classifier[0].in_features
    else:
        input_units = model.classifier.in_features

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.05)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if gpu:
        model.cuda()
    else:
        model.cpu()

    print_every = 40
    steps = 0
    train_dataloader,train_data = load_training_data(data_dir)
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(train_dataloader):
            steps +=1
            optimizer.zero_grad()
            if gpu:
                inputs = Variable(images.cuda())
                targets = Variable(labels.cuda())
            else:
                inputs = Variable(images)
                targets = Variable(labels)
            outputs = model.forward(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            running_loss +=loss.data[0]
            
        
            if steps%print_every == 0:
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every))           
                running_loss = 0
    
    train_data.class_to_idx = {v: k for k, v in train_data.class_to_idx.items()}
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model_arch': arch,
                  'classifier' : classifier,
                  'model': model,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint,'checkpoint.pth')

            
def load_training_data(data_dir):
    # TODO: Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return trainloader,train_data
# Call to main function to run the program
if __name__ == "__main__":
    main()