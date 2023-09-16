from unet_model import *

import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.transforms import ToTensor
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import logging
from accelerate import Accelerator
from tqdm import tqdm as std_tqdm
from functools import partial

def train(
        model,
        device,
        image_data,
        test_data,
        epochs: int = 30,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
    ):
        tqdm = partial(std_tqdm, dynamic_ncols=True)


        accelerator = Accelerator()

        
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(image_data)
            ), batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(test_data)
            ), batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

        for epoch in range(1, epochs+1):
            model.train()
            progress = tqdm(total=len(train_loader), desc="Training")
            # monitor training loss
            train_loss = 0.0
            
            ###################
            # train the model #
            ###################
            for data in train_loader:
                # _ stands in for labels, here
                # no need to flatten images
                images = torch.stack(data)[0]
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(images)
                # calculate the loss
                loss = criterion(outputs, images)
                # backward pass: compute gradient of the loss with respect to model parameters
                accelerator.backward(loss)
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item()
                progress.set_postfix({"Epoch": epoch, "train_loss": loss.item()})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
                progress.update()
                torch.cuda.empty_cache()           
            #print avg training statistics 
            train_loss = train_loss/len(train_loader)
            logging.info('Epoch: {} \tTraining Loss: {:.6f}'.format(
               epoch, 
               train_loss
               ))
            model.eval()
            
            progress = tqdm(total=len(test_loader), desc="Testing")
            with torch.no_grad():
                test_loss = 0.0
                for data in test_loader:
                    images = torch.stack(data)[0]
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = model(images)
                    # calculate the loss
                    loss = criterion(outputs, images)
                    test_loss += loss.item()
                    progress.set_postfix({"test_loss": loss.item()})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
                    progress.update()
                    torch.cuda.empty_cache()
                # print avg testing statistics 
                test_loss = test_loss/len(test_loader)
                logging.info('Epoch: {} \tTesting Loss: {:.6f}'.format(
                   epoch, 
                   test_loss
                   ))
        
        torch.save(model.state_dict(), 'unet.pt')


def test(
    model,
    device,
    test_data,
    batch_size: int = 4,
):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
    # monitor testing loss
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    ###################
    # test the model #
    ###################
    for data in test_loader:
        images = data
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # update running training loss
        test_loss += loss.item()*images.size(0)
            
    # print avg testing statistics 
    test_loss = test_loss/len(test_loader)
    print('Testing Loss: ' + f'{test_loss}')


def hist_img_to_tiles(hist_img, img_dim, img_stride):
    res = []
    for i in range(0, hist_img.shape[0]-img_dim, img_stride):
        for j in range(0, hist_img.shape[1]-img_dim, img_stride):
            res.append(hist_img[i:i+img_dim,j:j+img_dim,])
    return np.array(res)

def test_pretrained():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    image1 = np.load("rna_autoenc_c50400835s5_pos22_8bit.npy")
    images = [image1]
    tiles = [hist_img_to_tiles(img,(1<<5)-1,(1<<4)-1) for img in images]
    test_images_dataset = np.concatenate(tiles, axis=0)
    model = UNET(n_channels=test_images_dataset.shape[1])
    model.load_state_dict(torch.load('unet.pt'))
    model.to(device=device)
    with torch.no_grad():
        model.eval()
        test(model,device,test_images_dataset)
        
if __name__ == '__main__':  
    args = sys.argv
    if args[1] == 'train':
        image1 = np.load("rna_autoenc_c50400835s5_pos2_8bit.npy")
        image2 = np.load("rna_autoenc_c50400835s5_pos206_8bit.npy")
        images = [image1]
        tiles = [hist_img_to_tiles(img,(1<<5)-1,(1<<3)-1) for img in images]
        train_images_dataset = np.concatenate(tiles, axis=0)

        image1 = np.load("rna_autoenc_c50400835s5_pos22_8bit.npy")
        images = [image1]
        tiles = [hist_img_to_tiles(img,(1<<5)-1,(1<<3)-1) for img in images]
        #print(tiles[0].shape)
        test_images_dataset = np.concatenate(tiles, axis=0)


        latent_dim = 31
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        model = UNET(n_channels=train_images_dataset.shape[1], latent_dim=latent_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        model.to(device=device)
        train(model, device=device, image_data=train_images_dataset, test_data=test_images_dataset)
    elif args[1] == 'test':
        test_pretrained()

   # image1 = np.load("rna_autoenc_c50400835s5_pos22_8bit.npy")
   # images = [image1]
   # tiles = [hist_img_to_tiles(img,(1<<5)-1,(1<<4)-1) for img in images]
   # test_images_dataset = np.concatenate(tiles, axis=0)
   # model = UNET(n_channels=test_images_dataset.shape[1])
   # model.load_state_dict(torch.load('unet.pt'))
   # model.to(device=device)
   # model.eval()
   # test(model,test_images_dataset)
