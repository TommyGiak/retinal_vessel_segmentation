# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:51:50 2023

@author: FraTommaso 
LOL
"""

# U-Net home made for image segmentation

import matplotlib.pyplot as plt
from torchvision import transforms, datasets, utils
import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' #define computations on GPU if available

torch.manual_seed(0) # random seed for reproducibility



#%% data preprocessing for training

# transforamtions to apply a priori to the datasets
tr_data_rgb = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])]) # resized with bilinear interpolation to power of 2 for simplicity

tr_edge_target = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

train_folder = './datasets/training/images' # path where to find the rgb data (training)
data = datasets.ImageFolder(root=train_folder,transform=tr_data_rgb)

edge_folder = './datasets/training/edges' # path where to find the 'edge' data (training)
edges = datasets.ImageFolder(root=edge_folder,transform=tr_edge_target)

target_folder = './datasets/training/1st_manual' # path where to find the target data (training)
target = datasets.ImageFolder(root=target_folder,transform=tr_edge_target)

# the next part of code defines the data tensor with shape [n. images, channels, h, w]
# where channels = 4 (rgb + edge), and (h,w) = 512 (resized images)
tot_data = torch.vstack((data[0][0],edges[0][0])).unsqueeze(0)
tot_target = target[0][0].unsqueeze(0)

for i,imag in enumerate(data):
    
    sum_layers = torch.vstack((imag[0],edges[i][0]))
    tot_data = torch.vstack((tot_data,sum_layers.unsqueeze(0)))
    tot_target = torch.vstack((tot_target,target[i][0].unsqueeze(0)))

tot_data = tot_data[1:].to(device) # important .to(device) to move the tensors on the GPU, from now you'll see different .to(device) :)
data_edges = tot_data.detach() # the data is registered with and without the edge layer
data_no_edges = data_edges[:,:3].detach() # data without the edge layer
del(tot_data,sum_layers,edges,edge_folder,target_folder,train_folder,data)

tot_target = tot_target[1:].to(device)
target = tot_target.detach()
del(tot_target)
target[target>=0.05] = 1. # thresholds for the target (we want a tensor with only 0 or 1)
target[target<0.05] = 0. # due to bilinear interpolation in resize same of the pixels colud be different from 0 or 1

# now we have the variables 'data' and 'target' that contains all we need for the training


#%% data test_last_five
# the same code of above but this time its define the test data (last five images of trainig dataset)
# I will not repeat the comments

tr_data_rgb = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])])

tr_edge_target = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

train_folder = './datasets/training/images_test'
data_test = datasets.ImageFolder(root=train_folder,transform=tr_data_rgb)

edge_folder = './datasets/training/edges_test'
edges_test = datasets.ImageFolder(root=edge_folder,transform=tr_edge_target)

target_folder = './datasets/training/1st_manual_test'
target_test = datasets.ImageFolder(root=target_folder,transform=tr_edge_target)

tot_data = torch.vstack((data_test[0][0],edges_test[0][0])).unsqueeze(0)
tot_target = target_test[0][0].unsqueeze(0)

for i,imag in enumerate(data_test):
    
    sum_layers = torch.vstack((imag[0],edges_test[i][0]))
    tot_data = torch.vstack((tot_data,sum_layers.unsqueeze(0)))
    tot_target = torch.vstack((tot_target,target_test[i][0].unsqueeze(0)))

tot_data = tot_data[1:].to(device)
data_test_edges = tot_data.detach() # as before test data is registered with and without the edge layer
data_test_no_edges = data_test_edges[:,:3].detach() # test data without edge layers
del(tot_data,sum_layers,edges_test,edge_folder,target_folder,train_folder,data_test)

tot_target = tot_target[1:].to(device)
target_test = tot_target.detach()
del(tot_target)
target_test[target_test>=0.05] = 1.
target_test[target_test<0.05] = 0.

# now we have the variables 'data_test' and 'target_test' that contains all we need for the testing


#%% data test_no_groung
# this part is again the preprocessing of above but this time for the test dataset of the test folder WITHOUT the ground truth (no manual segmentation data) ;)

tr_data_rgb = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])])

tr_edge_target = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

test_folder = './datasets/test/images'
test = datasets.ImageFolder(root=test_folder,transform=tr_data_rgb)

edge_folder = './datasets/test/edges'
edges = datasets.ImageFolder(root=edge_folder,transform=tr_edge_target)

tot_test = torch.vstack((test[0][0],edges[0][0])).unsqueeze(0)

for i,imag in enumerate(test):
    
    sum_layers = torch.vstack((imag[0],edges[i][0]))
    tot_test = torch.vstack((tot_test,sum_layers.unsqueeze(0)))

tot_test = tot_test[1:].to(device)
test_edges = tot_test.detach() # again with edges
test_no_edges = test_edges[:,:3].detach() # without edges
del(tot_test,sum_layers,edges,imag,test_folder,test)

# now there is the variable 'test' (and no more) to see how the model perform on the test images of the dataset if someone want to 



#%% u-net with edges
# two U-Nets ar defined, one for the data with the edge layer and the other with only rgb channels

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet_for_a_new_hope_hard(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(4, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x) # size: 32 x 512 x 512
        enc2 = self.encoder2(F.max_pool2d(enc1, 2)) # size: 64 x 256 x 256
        enc3 = self.encoder3(F.max_pool2d(enc2, 2)) # size: 128 x 128 x 128
        enc4 = self.encoder4(F.max_pool2d(enc3, 2)) # size: 256 x 64 x 64

        # Decoding path with skip connections
        dec_3 = self.decoder1(enc4) # size: 128 x 128 x 128
        dec_3 = self.lrelu(dec_3)
        dec_3 = torch.cat([enc3, dec_3], dim=1) # size: 256 x 128 x 128

        dec_2 = self.decoder2(dec_3) # size: 64 x 256 x 256
        dec_2 = self.lrelu(dec_2)
        dec_2 = torch.cat([enc2, dec_2], dim=1) # size: 128 x 256 x 256

        dec_1 = self.decoder3(dec_2) # size: 32 x 512 x 512
        dec_1 = self.lrelu(dec_1)
        dec_1 = torch.cat([enc1, dec_1], dim=1) # size: 64 x 512 x 512

        final_output = self.final_conv(dec_1) # size: 1 x 512 x 512

        return F.sigmoid(final_output)



#%% u-net without edges

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet_but_no_edges(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(3, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x) # size: 32 x 512 x 512
        enc2 = self.encoder2(F.max_pool2d(enc1, 2)) # size: 64 x 256 x 256
        enc3 = self.encoder3(F.max_pool2d(enc2, 2)) # size: 128 x 128 x 128
        enc4 = self.encoder4(F.max_pool2d(enc3, 2)) # size: 256 x 64 x 64

        # Decoding path with skip connections
        dec_3 = self.decoder1(enc4) # size: 128 x 128 x 128
        dec_3 = self.lrelu(dec_3)
        dec_3 = torch.cat([enc3, dec_3], dim=1) # size: 256 x 128 x 128

        dec_2 = self.decoder2(dec_3) # size: 64 x 256 x 256
        dec_2 = self.lrelu(dec_2)
        dec_2 = torch.cat([enc2, dec_2], dim=1) # size: 128 x 256 x 256

        dec_1 = self.decoder3(dec_2) # size: 32 x 512 x 512
        dec_1 = self.lrelu(dec_1)
        dec_1 = torch.cat([enc1, dec_1], dim=1) # size: 64 x 512 x 512

        final_output = self.final_conv(dec_1) # size: 1 x 512 x 512

        return F.sigmoid(final_output)


#%% model definition

# by default the heavier model is defined
edges = input('If you want to evaluate/train the model WITH edges type yes, for no edges type no: ')
if edges == 'yes': # to define the model with the edge layers
    model = UNet_for_a_new_hope_hard().to(device) # here the model is defined with random parameters and it is moved on the GPU
    print('Edge model is defined')
elif edges == 'no': # model without the edges (only rgb)
    model = UNet_but_no_edges().to(device)
    print('NO-edge model is defined')
else:
    raise Warning(f'\'{edges}\' is not a possible choice, you sinner! No model is defined :(')
       

optim = torch.optim.Adam(model.parameters(), lr=1e-3) # ADAM optimizer for the gradient descent

loss = nn.BCELoss().to(device) # binary cross entropy loss for binary pixel-wise classification (vessel / not-vessel)
lossi = []


#%% load the beautiful model
# if a model with the name written in the 'path_unet' variable is found, it will be loaded (in the file there are only the parameters)
# ATTENTION: the parameters must be loaded on the right model (heavy or light) if not it will not work :(

path_unet = './models/unet_1500.pt'

try:
    if torch.cuda.is_available():
        print(model.load_state_dict(torch.load(path_unet)))
    else:
        print(model.load_state_dict(torch.load(path_unet, map_location=torch.device('cpu'))))
except:
    print('No previous model to load was found! The model still have random weights :(')


#%% training with edges
# beware: the training without a good GPU is extremely slow

model.train()

num_epochs = 500 # number of time the model will see the entire training data

train = input('If you want to start the training type yes, otherwise it will not start!')

if train == 'yes': 
    print('Training is starting...')
    name = model.__class__.__name__
    
    if name == 'UNet_for_a_new_hope_hard': # train the model with edges
        for epoch in range(num_epochs):
            
            # the next three lines asre used to shuffle date after each epoch
            index = torch.randperm(data_edges.shape[0])
            data_edges = data_edges[index]
            target = target[index] 
            
            for i in range(6): # the training data is divided in 6 batches
                # some theory ;) 
                model.zero_grad()
        
                output = model(data_edges[i*10:(i+1)*10])
                
                err = loss(output,target[i*10:(i+1)*10])
                err.backward()
                lossi.append(err.item())
                
                optim.step()
                
            print(f'Epoch: {epoch+1}/{num_epochs}, loss: {lossi[-1]:.4f}') # print the current loss at the end of each epoch
    
    elif name == 'UNet_but_no_edges': # train the model without edges
        for epoch in range(num_epochs):
            
            # the next three lines asre used to shuffle date after each epoch
            index = torch.randperm(data_no_edges.shape[0])
            data_no_edges = data_no_edges[index]
            target = target[index] 
            
            for i in range(6): # the training data is divided in 6 batches
                # some theory ;) 
                model.zero_grad()
        
                output = model(data_no_edges[i*10:(i+1)*10])
                
                err = loss(output,target[i*10:(i+1)*10])
                err.backward()
                lossi.append(err.item())
                
                optim.step()
                
            print(f'Epoch: {epoch+1}/{num_epochs}, loss: {lossi[-1]:.4f}') # print the current loss at the end of each epoch
    

#%% plot loss
# nothing it's just a function to plot the loss over the epochs

import numpy as np

def plot_loss(lossi : list, mean = 6, tit = None) -> None:

    lossi = np.array(lossi)
    y = lossi.reshape(-1,mean).mean(axis=1)
    x = np.linspace(1, len(y), num=len(y))
    fig, ax = plt.subplots()
    ax.plot(x,y)
    if tit is None:
        ax.set_title(f'Mean of {mean} losses steps')
    else:
        ax.set_title(tit)
    ax.set_ylabel('loss')
    ax.set_xlabel(f'steps/{mean}')
    ax.set_yscale('log')
    plt.show()
    pass

plot_loss(lossi)   
    


#%% sad human evaluation
# plot the input end output as images to see the progress

# numpy and matplotlib are quite annoing and they require the images as tensors of shape [h,w,channels] so we need to swap the axes (torchvision works with [channels,h,w])
name = model.__class__.__name__
if name == 'UNet_for_a_new_hope_hard':
    image = torch.swapaxes(torch.swapaxes(data_test_edges[0,:3],0,1),1,2)
elif name == 'UNet_but_no_edges':
    image = torch.swapaxes(torch.swapaxes(data_test_no_edges[0,:3],0,1),1,2)

plt.imshow(image.cpu(),cmap='gray')
plt.show()

model.eval()

with torch.no_grad():
    if name == 'UNet_for_a_new_hope_hard':
        plt.imshow(model(data_test_edges[0:1]).cpu().squeeze(), cmap='gray') # no axes swap because is a graylevel image of shape [h,w] (no channels since is only one)
    elif name == 'UNet_but_no_edges':
        plt.imshow(model(data_test_no_edges[0:1]).cpu().squeeze(), cmap='gray')
    plt.show()
    

#%% save the wonderful model
#ATTENTION: if you don't want to overwrite with previous models you must change the name

path_unet = './models/unet.pt'
fatal_decision = ''

print('Do you really want to save the model? Be careful with overwritings!')
fatal_decision = input('Type yes or no se if you want to save it or not: ')

if fatal_decision == 'yes':
    torch.save(model.state_dict(), path_unet)
    print(f'Model saved, path: {path_unet}')
elif fatal_decision == 'no':
    print('The model will not be saved!')
else:
    print(f'Oh funny, but "{fatal_decision}" is not an accaptable answer! For revenge I will not save the model :P')



#%% output test, we are gonna save this IMAGES
# it just save the output of the model as tiff images (rgb), so a bit of postprocessing is needed

model.eval()

resize = transforms.Resize([584,565])

path_results = './datasets/results/'
name = model.__class__.__name__

with torch.no_grad():
    
    if name == 'UNet_for_a_new_hope_hard':
        for i, imag in enumerate(model(data_test_edges[:5])):
            utils.save_image(resize(imag), path_results+f'result_{i}.tiff')
    elif name == 'UNet_but_no_edges':
        for i, imag in enumerate(model(data_test_no_edges[:5])):
            utils.save_image(resize(imag), path_results+f'result_{i}.tiff')






