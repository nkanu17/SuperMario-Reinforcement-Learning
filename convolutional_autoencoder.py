import numpy as np
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision import utils
import matplotlib.pyplot as plt

import MarioData as md
from torch.utils.data import DataLoader
from skimage import img_as_float

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time
import torch.nn as nn



class Convolutional_AutoEncoder(torch.nn.Module):
    # Input is (3, 128, 128)

    def __init__(self):
        super(Convolutional_AutoEncoder, self).__init__()
        ## encoder layers ##
        
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d( 3, 16, 3, padding = 1)  
        # conv layer (depth from 16 --> 9), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 9, 3, padding = 1)
        
        ## decoder layers ##
        ## a kernel of 2 
        # no stride
        
        self.t_conv1 = nn.ConvTranspose2d(9, 16, 1)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 1)


    def forward(self, x):
        
        ## encode ##
        # add hidden layers with relu activation function       
        x = F.relu(self.conv1(x))

        # add second hidden layer
        x = F.relu(self.conv2(x))
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))


        return x

    @staticmethod
    def createLossAndOptimizer(net, learning_rate=0.001):
        #Loss function using MSE due to the task being reconstruction
        loss = torch.nn.MSELoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)


def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = net.createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs = data['image'].to(device)
            
            #Wrap them in a Variable object
            inputs = Variable(inputs)

            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)

            loss_size = loss(outputs, inputs)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()

            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                # print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                #         epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                print(i)
                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), running_loss / print_every, 0, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(val_loader, 0):
            #Get inputs
            inputs = data['image'].to(device)
            #Wrap tensors in Variables
            inputs = Variable(inputs)
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, inputs)
            total_val_loss += val_loss_size.data.item()
            
        print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader),0))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def show_images(images):
    """Show image with Mario annotated for a batch of samples."""
    img = images
    grid = utils.make_grid(img)
    plt.imshow(img_as_float(grid.numpy().transpose((1, 2, 0))))         
    plt.axis('off')
    plt.ioff()
    plt.show()



def testNet(net, test_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            print(i)
            inputs = data['image'].to(device)
            
            inputs = Variable(inputs)
            
            outputs = net(inputs)
                        
            final_inputs = inputs
            final_outputs = outputs            
            
            #show real images
            show_images(inputs)
            #show reconstructed images
            show_images(outputs)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ********************************
    # Dataset stuff
    transform2apply = transforms.Compose([
                                            transforms.Resize((128,128)),
                                            transforms.ToTensor()
                                        ])
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    dataset = md.DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    use_cuda = True

    # Creating data indices for training, validation and test splits:
    dataset_size = len(dataset)
    n_test = int(dataset_size * 0.05)
    n_train = dataset_size - 2 * n_test

    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:(n_train + n_test)]
    test_indices = indices[(n_train + n_test):]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    print("Train size: {}".format(len(train_sampler)))
    print("Validation size: {}".format(len(valid_sampler)))
    print("Test size: {}".format(len(test_sampler)))

    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler, num_workers=0, drop_last = True)
    validation_loader = DataLoader(dataset, batch_size=128, sampler=valid_sampler, num_workers=0, drop_last = True)
    test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=0, drop_last = True)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    CAE = Convolutional_AutoEncoder().to(device)
    print(CAE)
    
    
    trainNet(CAE, train_loader, validation_loader, n_epochs=1, learning_rate=0.001, device=device)

    testNet(CAE, test_loader, device=device)

    print("Done!")
