import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch_cv_wrapper.dataloader.albumentation import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import notebook
from PIL import Image
import os
import requests
import zipfile
from io import BytesIO
import glob
import csv
import numpy as np
import random

class Cifar10DataLoader:
    def __init__(self, config):
        self.config = config
        self.augmentation = config['data_augmentation']['type']
        
    def calculate_mean_std(self):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

    def classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], None
        
    def get_dataloader(self): 
        
        cifar_albumentation = eval(self.augmentation)()
        mean,std = self.calculate_mean_std()
        
        train_transforms, test_transforms = cifar_albumentation.train_transform(mean,std),cifar_albumentation.test_transform(mean,std)
                                                                              
        trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)  
            
        testset  = datasets.CIFAR10(root='./data', train=False,
                                             transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                      batch_size=self.config['data_loader']['args']['batch_size'], 
                                                      shuffle=True,
                                                      num_workers=self.config['data_loader']['args']['num_workers'], 
                                                      pin_memory=self.config['data_loader']['args']['pin_memory'])
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                     batch_size=self.config['data_loader']['args']['batch_size'],  
                                                     shuffle=False,
                                                     num_workers=self.config['data_loader']['args']['num_workers'], 
                                                     pin_memory=self.config['data_loader']['args']['pin_memory'])
        return self.train_loader,self.test_loader


class TinyImageNetDataLoader:

    def __init__(self, config=None):
        self.config = config
        if config is not None:
            self.augmentation = config['data_augmentation']['type']
        
    def calculate_mean_std(self):
        mean,std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        return mean, std

    def classes(self):
        id_dict = {}
        all_classes = {}
        for i, line in enumerate(open( 'data/tiny-imagenet-200/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        
        result = {}
        class_id={}
        for i, line in enumerate(open( 'data/tiny-imagenet-200/words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (all_classes[key].replace('\n','').split(",")[0])
            class_id[key] = (value,all_classes[key])
            
        return result,class_id  
        
    def get_dataloader(self): 
        
        tinyimagenet_albumentation = eval(self.augmentation)()
        #cifar_albumentation = CIFAR10Albumentation()
        mean,std = self.calculate_mean_std()
        
        train_transforms, test_transforms = tinyimagenet_albumentation.train_transform(mean,std),tinyimagenet_albumentation.test_transform(mean,std)
                                                                              
        trainset = TinyImageNet(root='./data', train=True,download=True, transform=train_transforms) 
        # train_len = int(train_split*len(trainset))
        # valid_len = len(trainset) - train_len
        #imagenet_trainset, imagenet_valset = torch.utils.data.random_split(trainset, [train_len, valid_len])
            
        testset  = TinyImageNet(root='./data', train=False,transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                      batch_size=512, 
                                                      shuffle=True,
                                                      num_workers=2, 
                                                      pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                     batch_size=512,  
                                                     shuffle=False,
                                                     num_workers=2, 
                                                     pin_memory=True)
        return self.train_loader,self.test_loader


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None,  download=False,train_split=0.7):
        
        self.root = root
        self.transform = transform
        self.data_dir = 'tiny-imagenet-200'

        if download and (not os.path.isdir(os.path.join(self.root, self.data_dir))):
            self.download()

        self.image_paths = []
        self.targets = []

        _,class_id = TinyImageNetDataLoader().classes()
        
        # train images
        train_path = os.path.join(self.root, self.data_dir, 'train')
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, 'images')
            for image in os.listdir(train_images_path):
                if image.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(train_images_path, image))
                    self.targets.append(class_id[class_dir][0])

        # val images
        val_path = os.path.join(self.root, self.data_dir, 'val')
        val_images_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                self.image_paths.append(os.path.join(val_images_path, line[0]))
                self.targets.append(class_id[line[1]][0])
                
        self.indices = np.arange(len(self.targets))

        random_seed=1
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        split_idx = int(len(self.indices) * train_split)
        self.indices = self.indices[:split_idx] if train else self.indices[split_idx:]

    def download(self):
        if (os.path.isdir("./data/tiny-imagenet-200")):
            print('Images already downloaded...')
            return
        r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
        print('Downloading TinyImageNet Data')
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path='./data/')
        zip_ref.close()


    def __getitem__(self, index):
        
        image_index = self.indices[index] 
        filepath = self.image_paths[image_index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.targets[image_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)
