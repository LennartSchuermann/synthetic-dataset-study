import glob
import cv2
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid

classes = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break
    
class SyntheticExpressionDataset(Dataset):
    def __init__(self, train : bool):
        folder_path = ""
        if train: folder_path= "train" 
        else: folder_path="validation"
        
        print("Initializing Synthetic " + folder_path + " Dataset...")
        
        self.imgs_path = "dataset_synthetic/" + folder_path + "/"
        sub_folders = [name for name in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, name))]
        self.data = []
        
        for i in range(len(sub_folders)):
            
            path = self.imgs_path + sub_folders[i] + "/"
            file_list = glob.glob(path + "*")
            
            for img_path in file_list:
                self.data.append([img_path, sub_folders[i]])

        self.class_map = {"angry" : 0, 
                        "disgust": 1, 
                        "fear": 2, 
                        "happy": 3, 
                        "neutral": 4,
                        "sad": 5,
                        "surprise": 6}
        self.img_dim = (48, 48)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ]) 
        img_tensor = transform(img)
        
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        
        return img_tensor, class_id

class ExpressionDataset(Dataset):
    def __init__(self, train : bool):
        folder_path = ""
        if train: folder_path= "train" 
        else: folder_path="validation"
        
        print("Initializing " + folder_path + " Dataset...")
        
        self.imgs_path = "dataset_base/" + folder_path + "/"
        sub_folders = [name for name in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, name))]
        self.data = []
        
        for i in range(len(sub_folders)):
            
            path = self.imgs_path + sub_folders[i] + "/"
            file_list = glob.glob(path + "*")
            
            for img_path in file_list:
                self.data.append([img_path, sub_folders[i]])
                    
        self.class_map = {"angry" : 0, 
                        "disgust": 1, 
                        "fear": 2, 
                        "happy": 3, 
                        "neutral": 4,
                        "sad": 5,
                        "surprise": 6}
        self.img_dim = (48, 48)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        
        transform = transforms.Compose([ 
            transforms.ToTensor() 
        ]) 
        img_tensor = transform(img)
        
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        
        return img_tensor, class_id

# Test The DataLoaders
#if __name__ == "__main__":
    #dataset = SyntheticExpressionDataset(train=True)
    #data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Display Random Batch (image & label)
    #show_batch(data_loader)
