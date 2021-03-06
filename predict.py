import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import time
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from collections import OrderedDict
import os.path
import PIL
from PIL import Image
import numpy as np
import json


################### Input #######################

parser = argparse.ArgumentParser()
  
parser.add_argument('-im', nargs='?', default = 'flowers/test/87/image_05466.jpg')
parser.add_argument('-checkpoint', nargs="?", default = "savedir/checkpoint.pth")
parser.add_argument('-device', nargs="?", default = 'gpu')


args = parser.parse_args()
im = args.im; device = args.device; checkpoint = args.checkpoint
if device == 'gpu':
    device = 'cuda'

############### Cat to name mapping #######################    
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
############### Define functions preprocess, predict ans plt #######################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    tens_im = process_image(image_path)    
    with torch.no_grad():
        tens_im = torch.FloatTensor(tens_im)
        tens_im.unsqueeze_(0)
        probabilities = torch.exp(model.forward(tens_im.to(device)))    
        top_p, top_class = probabilities.topk(5, dim=1)
        top_p = top_p.to("cpu"); top_class = top_class.to("cpu");
        top_p_np = top_p.numpy()[0]; top_class_np = top_class.numpy()[0]
        
        #convert to real classes with class_to_idx
        inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
        top_class_afterclasstoidx = [inv_class_to_idx.get(key) for key in top_class_np]
        
        #cat_to_name
        top_class_names = []
        for x in range(5):
            top_class_names.append(cat_to_name[str(top_class_afterclasstoidx[x])])
               
        return top_p_np, top_class_afterclasstoidx, top_class_names 

    
    
##################### load checkpoint #####################    

if os.path.isfile(checkpoint):
    print ("Checkpoint file exist. Loading checkpoint.")
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    arch = checkpoint['arch']

    model = getattr(models, arch)(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('drop', nn.Dropout(p=0.5)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
           
else:
    print("No checkpoint found, can't predict.")    

    
#######################            #################################    
    
print(" ")
print("Flower file: ", im)


top_p, top_class, top_class_names = predict(im, model)


print("Class numbers: ",top_class)
print("Class names:   ",top_class_names)
print("Probabilities: ",np.round(top_p))
print(" ")




    
