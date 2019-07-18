import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform, color, img_as_float, img_as_int
from PIL import Image
from scipy import misc

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils

class DatasetMario(Dataset):
    def __init__(self, file_path, csv_name, transform_in=None):
        self.data = pd.read_csv(file_path+"/"+csv_name)
        self._path = file_path+"/"
        self.transform_in = transform_in

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        image_file = self._path + self.data.iloc[index, 0]#.values.astype(np.uint8).reshape((1, 28, 28))
        state_fn = self._path + self.data.iloc[index, 1]

        # Get image
        img_rgba = Image.open(image_file)
        image_data = img_rgba.convert('RGB')

        # Get data from state.json
        with open(state_fn) as data_file:
            data = data_file.read()
            state_data = json.loads(data)

        # Get controller 1 info
        if "controller" in state_data:
            # {'A', 'B', 'down', 'left', 'right', 'select', 'start', 'up': False}
            control_data = np.asarray(list(state_data['controller'][0].values()), dtype=np.int)
            control_data = np.asarray([1])#np.asarray(list(range(0,len(control_data)))) * control_data
        else:
            control_data = np.zeros(8, dtype=np.int)

        # Get mario info
        if "mario" in state_data:
            mario_handler = state_data['mario']
            mpos = mario_handler["pos"]
            x1,y1,x2,y2 = mpos[0]-1,mpos[1]-8,mpos[2],mpos[3]-8 # For all items, this offsets work
            mario_handler['pos'] = [x1,y1,x2,y2]
            mario_handler['pos'] = np.asarray(mario_handler['pos'], dtype=np.int).reshape((2, 2))

            coins = mario_handler['coins']#
            level = mario_handler['level']#
            lives = mario_handler['lives']#
            pos1, pos2 = mario_handler['pos']
            state = mario_handler['state']#
            world = mario_handler['world']#
            tmp_array = [state, lives, world, level, coins, pos1[0], pos1[1], pos2[0], pos2[1]]
            mario_data = np.asarray(tmp_array)
        else:
            mario_data = np.zeros(9, dtype=np.int)

        # Get enemy info
        #enemy_data = []
        enemy_data = np.asarray([0], dtype=np.int64)
        for i in range(0,5):
            key = "enemy"+str(i)
            if key in state_data:
                enemy_data = np.asarray([1], dtype=np.int64)
                #enemy_data.append(np.asarray(state_data["enemy"+str(i)], dtype=np.int))
            # else:
                # enemy_data.append(None)

        if self.transform_in is not None:
            image_data = self.transform_in(image_data)

        sample = {'image': image_data, 'state': torch.from_numpy(control_data).type(torch.long), 'mario': torch.from_numpy(mario_data), 'enemy': enemy_data}

        return sample


class Rescale(object):
    # From Pytorch tutorials
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h 
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        # sample['image'] = transform.resize(image, (new_h, new_w), anti_aliasing=True)
        sample['image'] = misc.imresize(image, (new_h, new_w), 'nearest')

        # sample['mario'] = sample['mario']['pos'] * [new_w / w, new_h / h]

        # if sample['enemy']:
            # sample['enemy']['pos'] = sample['enemy']['pos'] * [new_w / w, new_h / h]

        return sample

class ToTensor(object):

    def __call__(self, sample):
        image = torch.from_numpy(sample['image'].transpose((2,0,1)))
        state = torch.from_numpy(sample['state']).float()
        mario = torch.from_numpy(sample['mario']).float()
        # if 'mario' in sample:
        #     coins = sample['mario']['coins']#
        #     level = sample['mario']['level']#
        #     lives = sample['mario']['lives']#
        #     pos1, pos2 = sample['mario']['pos']
        #     state = sample['mario']['state']#
        #     world = sample['mario']['world']#
        #     tmp_array = [state, lives, world, level, coins, pos1[0], pos1[1], pos2[0], pos2[1]]
        #     tmp_array = np.asarray(tmp_array, dtype=np.int)
        #     sample['mario'] = torch.from_numpy(tmp_array).float()
        # # if 'enemy' in sample:
        # #     sample['enemy'] = torch.from_numpy(sample['enemy'])

        return {'image': image, 'state': state, 'mario': mario}


def show_data(ax, image, state, mario, enemy):
    """Show image with info from state"""
    plt.imshow(image.permute(1, 2, 0).numpy())

    # if mario:
    #     # mpos = state["mario"]["pos"]
    #     # x1,y1,x2,y2 = mpos[0]-1,mpos[1]-8,mpos[2],mpos[3]-8 # For all items, this offsets work
    #     pt1, pt2 = mario["pos"]
    #     x1,y1,x2,y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    #     w,h = x2-x1, y2-y1
    #     plt.plot(x1+w/2,y2-h/2, '*y')
    #     rect = patches.Rectangle((x1,y1),w,h,linewidth=3,edgecolor='g',facecolor='none')
    #     ax.add_patch(rect)

    # for e in enemy:
    #     if e is not None:
    #         print("There's an enemy, implement plot")

    plt.pause(0.001)  # pause a bit so that plots are updated

# Helper function to show a batch
def show_data_batch(sample_batched):
    """Show image with Mario annotated for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(img_as_float(grid.numpy().transpose((1, 2, 0))))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    # ***********************
    # DEMO
    # ***********************

    # transform2apply = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    # mario_dataset = DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)
    # sample = mario_dataset.__getitem__(0)
    # print(sample)

    # fig = plt.figure()
    # for i in range(len(mario_dataset)):
    #     sample = mario_dataset[i]
    #     print(i, sample['image'].size(), sample['state'].size())

    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     show_data(ax, **sample)

    #     if i == 3:
    #         plt.show()
    #         break

    # print("Done Displaying Images!")

    # ***********************
    # ToTensor demo
    # ***********************
    transform2apply = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transformed_dataset = DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['state'].size())
        if i == 3:
            break

    print("Done Transforming to Tensor!")

    # ***********************
    # Dataloader demo
    # ***********************
    # From: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['state'].size(), sample_batched['mario'].size())

        # observe 0th batch and stop.
        if i_batch == 0:
            plt.figure()
            show_data_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
    