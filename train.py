########################### import ###########################
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




########################### parser ###########################
parser = argparse.ArgumentParser()
  
parser.add_argument('-data_dir', nargs='?', default = 'flowers')
parser.add_argument('-save_dir', nargs="?", default = "savedir")
parser.add_argument('-arch', nargs="?", default = "vgg13")
parser.add_argument('-learning_rate', nargs="?", default = 0.001, type = float)
parser.add_argument('-hidden_units', nargs="?", default = 512, type = int)
parser.add_argument('-epochs', nargs="?", default= 10, type = int)
parser.add_argument('-gpu', nargs="?", default = 1, type = int)

args = parser.parse_args()

########################### startmessage ###########################

print("  ")
print("     Start the Flower training Module     ")
print("  ")

########################### loading data ###########################
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

########################### transforms ###########################
data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

data_transforms_test_valid = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


########################### Load the datasets with ImageFolder ###########################
image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_test_valid)
image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test_valid)


###########################  dataloaders ###########################
trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=32)


########################### Label mapping ###########################
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
########################### define model ########################### 

model = getattr(models, args.arch)(pretrained=True)

device = torch.device("cuda" if args.gpu == 1 else "cpu")

for param in model.parameters(): 
    param.requires_grad = False

model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss() 
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate *1/3)
                       
########################### loading checkpoint if availabe ########################### 


use_checkpoint = input("Use checkpoint? Press y ----> ")

if use_checkpoint == 'y': 
    if os.path.isfile(args.save_dir+"/checkpoint.pth"):
        print ("Checkpoint file exists. Loading checkpoint and continuing training.")

        checkpoint = torch.load(args.save_dir+"/checkpoint.pth", map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_passed = checkpoint['epochs_passed']
        loss = checkpoint['loss']
        steps = checkpoint['steps']
        valid_loss = checkpoint['valid_loss']
        accuracy = checkpoint['accuracy']
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        print("No checkpoint found, training from scratch")
else:
    print("Training from scratch")
    epochs_passed = 0
########################### training ###########################

if args.epochs <= epochs_passed:
    print("You already trained", epochs_passed , "epochs. Try more epochs.")
    exit()

for e in range(epochs_passed,args.epochs):
    if args.epochs < 4:
        optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate * 1/3)
    else:
        if e < 2:
            optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
        elif e < 5:
            optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate *2/3)
        else:
            optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate *1/3)
    steps = 0
    print_every = 10
    model.to(device)
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        model.train()
        steps += 1
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, args.epochs))
            
            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            model.eval () #switching to evaluation mode so that dropout is turned off
            
            # Turn off gradients for validation to save memory and computations
            with torch.no_grad():
                
                model.to(device)
    
                valid_loss = 0
                accuracy = 0
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()
                     
                
            print(
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validloader)*100),
                  "Learning rate: {:.10f}..".format(optimizer.param_groups[0]['lr']))
               
            running_loss=0
    epochs_passed = e+1

print("Training ended.")            


########################### Save model to checkpoint file ###########################

cp =  args.save_dir+"/checkpoint.pth"


torch.save({
            'epochs_passed': epochs_passed,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classifier': model.classifier,
            'loss': loss,
            'valid_loss':valid_loss,
            'accuracy': accuracy,
            'class_to_idx': image_datasets_train.class_to_idx,
            'arch' : args.arch,
            }, cp)  


print("Model and parameters saved in savedir/checkpoint.pth")