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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)

#%% data preprocessing

first_tr_data = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])]) # resized with bilinear interpolation to power of 2 for simplicity

first_tr_other = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

train_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/images'
data = datasets.ImageFolder(root=train_folder,transform=first_tr_data)

edge_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/edges'
edges = datasets.ImageFolder(root=edge_folder,transform=first_tr_other)

target_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/1st_manual'
target = datasets.ImageFolder(root=target_folder,transform=first_tr_other)

tot_data = torch.vstack((data[0][0],edges[0][0])).unsqueeze(0)
tot_target = target[0][0].unsqueeze(0)

for i,imag in enumerate(data):
    
    sum_layers = torch.vstack((imag[0],edges[i][0]))
    tot_data = torch.vstack((tot_data,sum_layers.unsqueeze(0)))
    tot_target = torch.vstack((tot_target,target[i][0].unsqueeze(0)))

tot_data = tot_data[1:].to(device)
data = tot_data.detach()
del(tot_data,sum_layers,edges,edge_folder,target_folder,train_folder)

tot_target = tot_target[1:].to(device)
target = tot_target.detach()
del(tot_target)
target[target>=0.05] = 1.
target[target<0.05] = 0.

#%% data test_last_five

first_tr_data = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])]) # resized with bilinear interpolation to power of 2 for simplicity

first_tr_other = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

train_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/images_test'
data_test = datasets.ImageFolder(root=train_folder,transform=first_tr_data)

edge_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/edges_test'
edges_test = datasets.ImageFolder(root=edge_folder,transform=first_tr_other)

target_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/training/1st_manual_test'
target_test = datasets.ImageFolder(root=target_folder,transform=first_tr_other)

tot_data = torch.vstack((data_test[0][0],edges_test[0][0])).unsqueeze(0)
tot_target = target_test[0][0].unsqueeze(0)

for i,imag in enumerate(data_test):
    
    sum_layers = torch.vstack((imag[0],edges_test[i][0]))
    tot_data = torch.vstack((tot_data,sum_layers.unsqueeze(0)))
    tot_target = torch.vstack((tot_target,target_test[i][0].unsqueeze(0)))

tot_data = tot_data[1:].to(device)
data_test = tot_data.detach()
del(tot_data,sum_layers,edges_test,edge_folder,target_folder,train_folder)

tot_target = tot_target[1:].to(device)
target_test = tot_target.detach()
del(tot_target)
target_test[target_test>=0.05] = 1.
target_test[target_test<0.05] = 0.

#%% data test_no_groung

first_tr_data = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512])]) # resized with bilinear interpolation to power of 2 for simplicity

first_tr_other = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize([512,512]),
                               transforms.Grayscale()])

test_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/test/images'
test = datasets.ImageFolder(root=test_folder,transform=first_tr_data)

edge_folder = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/test/edges'
edges = datasets.ImageFolder(root=edge_folder,transform=first_tr_other)

tot_test = torch.vstack((test[0][0],edges[0][0])).unsqueeze(0)

for i,imag in enumerate(test):
    
    sum_layers = torch.vstack((imag[0],edges[i][0]))
    tot_test = torch.vstack((tot_test,sum_layers.unsqueeze(0)))

tot_test = tot_test[1:].to(device)
test = tot_test.detach()
del(tot_test,sum_layers,edges,imag,test_folder)


#%% u-net pesante

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
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
        enc2 = self.encoder2(F.max_pool2d(enc1, 2)) # size: 63 x 256 x 256
        enc3 = self.encoder3(F.max_pool2d(enc2, 2)) # size: 128 x 128 x 128
        enc4 = self.encoder4(F.max_pool2d(enc3, 2)) # size: 256 x 64 x 64

        # Decoding path with skip connections
        dec_3 = self.decoder1(enc4) # size: 64 x 128 x 128
        dec_3 = self.lrelu(dec_3)
        dec_3 = torch.cat([enc3, dec_3], dim=1) # size: 128 x 128 x 128

        dec_2 = self.decoder2(dec_3) # size: 32 x 256 x 256
        dec_2 = self.lrelu(dec_2)
        dec_2 = torch.cat([enc2, dec_2], dim=1) # size: 64 x 256 x 256

        dec_1 = self.decoder3(dec_2) # size: 16 x 512 x 512
        dec_1 = self.lrelu(dec_1)
        dec_1 = torch.cat([enc1, dec_1], dim=1) # size: 32 x 512 x 512

        final_output = self.final_conv(dec_1)

        return F.sigmoid(final_output)
    

#%% u-net leggera

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet_for_a_new_hope_ez(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(4, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.encoder4 = DoubleConv(64, 128)

        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x) # size: 32 x 512 x 512
        enc2 = self.encoder2(F.max_pool2d(enc1, 2)) # size: 63 x 256 x 256
        enc3 = self.encoder3(F.max_pool2d(enc2, 2)) # size: 128 x 128 x 128
        enc4 = self.encoder4(F.max_pool2d(enc3, 2)) # size: 256 x 64 x 64

        # Decoding path with skip connections
        dec_3 = self.decoder1(enc4) # size: 64 x 128 x 128
        dec_3 = self.lrelu(dec_3)
        dec_3 = torch.cat([enc3, dec_3], dim=1) # size: 128 x 128 x 128

        dec_2 = self.decoder2(dec_3) # size: 32 x 256 x 256
        dec_2 = self.lrelu(dec_2)
        dec_2 = torch.cat([enc2, dec_2], dim=1) # size: 64 x 256 x 256

        dec_1 = self.decoder3(dec_2) # size: 16 x 512 x 512
        dec_1 = self.lrelu(dec_1)
        dec_1 = torch.cat([enc1, dec_1], dim=1) # size: 32 x 512 x 512

        final_output = self.final_conv(dec_1)

        return F.sigmoid(final_output)
    

#%% model definition

model = UNet_for_a_new_hope_ez().to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

loss = nn.BCELoss().to(device)
lossi = []

#%% training

model.train()

num_epochs = 5


for epoch in range(num_epochs):

    for i in range(6):
            
        model.zero_grad()
    
        output = model(data[i*10:(i+1)*10])
    
        err = loss(output,target[i*10:(i+1)*10])
        err.backward()
    
        optim.step()
    
        lossi.append(err.item())
        
    print(f'Epoch: {epoch+1}/{num_epochs}, loss: {lossi[-1]:.4f}')
    
#%% plot loss

import numpy as np

def plot_loss(lossi : list, mean = 1, tit = None) -> None:

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

image = torch.swapaxes(torch.swapaxes(data_test[0,:3],0,1),1,2)
plt.imshow(image.cpu(),cmap='gray')
plt.show()

model.eval()

with torch.no_grad():
    plt.imshow(model(data_test[0:1]).cpu().squeeze(), cmap='gray')
    plt.show()
    

#%% save the wonderful model
path_unet = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/models/unet.pt'

torch.save(model.state_dict(), path_unet)

#%% load the awful model
path_unet = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/models/unet.pt'

if torch.cuda.is_available():
    model.load_state_dict(torch.load(path_unet))
else:
    model.load_state_dict(torch.load(path_unet, map_location=torch.device('cpu')))


#%% output test, we are gonna save this IMAGES

model.eval()

resize = transforms.Resize([584,565])

path_results = '/Users/tommygiak/Desktop/retinal_vessel_segmentation/datasets/test/results/'

with torch.no_grad():
    
    for i, imag in enumerate(model(test)):
        
        utils.save_image(resize(imag), path_results+f'result_{i}.tiff')
    






