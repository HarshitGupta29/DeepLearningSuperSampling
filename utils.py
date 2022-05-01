import model 
from typing import Optional, Tuple, List
import numpy as np
import torch
import os
from PIL import Image


def batchify(lst: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """ Create array of batches of size batch_size
        All images in a batch should be stacked in the z dimension
    """
    count=0
    temp = []
    final =[]
    for arr in lst:
        if(count<batch_size):
            temp.append(arr)
            count+=1          
        else:
            final.append(np.stack(temp,axis=3))
            temp=[arr]
            count=1
    if temp:
        final.append(np.stack(temp,axis=3))
    return final


def padding(image: np.ndarray, pad_x: int, pad_y: int) -> np.ndarray:
    """ Add padding to image on both dimensions
    """
    return np.pad(image, ((pad_x, pad_x),(pad_y, pad_y), (0, 0)),mode='constant', constant_values=0)


def break_image(image: np.ndarray, dim_x: int, dim_y: int, brake = False) -> List[np.ndarray]:
    """ Return a list of images with dimensions dim_x and dim_y
        Take image and cut it into different pieces where each piece has given dimension
        If dimension is less than use padding
        If brake is True then don't cut the image, just return as it is
        For dimensions indexing starts from 1
    """
    if brake:
        return image
        
    temp = np.array([])
    rows, cols, height= np.shape(image)
    temp = image
    if rows%dim_y !=0:
        temp= np.pad(image, ((0,0),(0, abs(dim_y-rows%dim_y)), (0, 0)), mode='constant', constant_values=0)
    if cols%dim_x !=0:
        temp = np.pad(temp , ((0, abs(dim_x-cols%dim_x)),(0, 0), (0, 0)),mode='constant', constant_values=0)

    rows2, cols2, height2= np.shape(temp)

    rowsplits = rows2//dim_y
    colsplits = cols2//dim_x
    vals = np.vsplit(temp, rowsplits)
    final = []
    for k in vals:
        final.extend(np.hsplit(k, colsplits))
    return final

def load_data(input: List[np.ndarray], image_x:int, image_y: int, batch_size: int) -> List[np.ndarray]:
    """ Load the data and return list of input and output images

        Note: if input is None, then the input is taken from DIV2K_train_HR
    """
    if input:
        vals = input
    else:
        dirpath = 'DIV2K_train_HR/'
        vals = os.listdir(dirpath)
    final = []
    for pic in vals:
        img = Image.open(pic)
        arr = np.array(img)
        final.extend(break_image(arr, image_x, image_y))
    return batchify(final, batch_size)


def train(input: List[np.ndarray], target: List[np.ndarray], model: torch.nn.Module, dim_x, dim_y, batch_size):
    """ Create batches of input and target sequentially
        Feed input into the model
        Optimize and save the weights
        Note: if input is None, then in load_data the input is taken from DIV2K_train_HR
    """
    data= load_data(input, dim_x, dim_y, batch_size)
    final =[]
    for inp in input:
        final.append(model(inp))
    return final

if __name__ == '__main__':
    stuff  = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    stuff2 = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    stuff3 = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    stuff4 = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    stuff5 = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
    temp = batchify([stuff, stuff2, stuff3,stuff4,stuff5], 3)
    print(temp)


    #train(input, output, model.dlss)