import torchvision
from dataloader.load_data import Cifar10DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from utils import train as trn
from utils import test as tst
from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class TriggerEngine:
    def __init__(self, config):
        self.config = config
        self.cifar_dataset=Cifar10DataLoader(self.config)
        self.device = self.set_device()
        self.class_names = self.config['data_loader']['classes']
        self.writer = SummaryWriter()

        
    def dataloader(self):
        #Get dataloaders
        train_loader,test_loader = self.cifar_dataset.get_dataloader()
        return train_loader,test_loader
        
        
    def set_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device
        
    def run_experiment(self,model,train_loader,test_loader):
        
        model.to(self.device) 
        dropout=self.config['model_params']['dropout']
        epochs=self.config['training_params']['epochs']
        l2_factor = self.config['training_params']['l2_factor']
        l1_factor = self.config['training_params']['l1_factor']
        
        criterion = nn.CrossEntropyLoss() if self.config['criterion'] == 'CrossEntropyLoss' else F.nll_loss()
        opt_func = optim.Adam if self.config['optimizer']['type'] == 'optim.Adam' else optim.SGD
        lr = self.config['optimizer']['args']['lr']
        
        grad_clip = 0.1
            
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
        lrs=[]
            
        
        #optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.7,weight_decay=l2_factor)
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=l2_factor)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True,mode='max')
        scheduler = OneCycleLR(optimizer, max_lr=lr,epochs=epochs,steps_per_epoch=len(train_loader))

        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}:')
            trn.train(model, self.device, train_loader, optimizer,epoch, train_accuracy, train_losses, l1_factor,scheduler,criterion,lrs,self.writer,grad_clip)
            tst.test(model, self.device, test_loader,test_accuracy,test_losses,criterion)
            
            self.writer.add_scalar('Epoch/Train/train_loss', train_losses[-1], epoch)
            self.writer.add_scalar('Epoch/Test/test_loss', test_losses[-1], epoch)
            self.writer.add_scalar('Epoch/Train/train_accuracy', train_accuracy[-1], epoch)
            self.writer.add_scalar('Epoch/Train/test_accuracy', test_accuracy[-1], epoch)

            # if epoch > 20:
            #     scheduler.step(test_accuracy[-1])

            self.writer.flush()
        return (train_accuracy,train_losses,test_accuracy,test_losses)
        
    def save_experiment(self,model, experiment_name):
        print(f"Saving the model for {experiment_name}")
        torch.save(model, './saved_models/{}.pt'.format(experiment_name))
    
    def model_summary(self,model, input_size):
        result = summary(model, input_size=input_size)
        print(result)    
        
    def wrong_predictions(self,model,test_loader):
        wrong_images=[]
        wrong_label=[]
        correct_label=[]
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)        
                pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])  
      
                wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
            print(f'Total wrong predictions are {len(wrong_predictions)}')
            
            self.plot_misclassified(wrong_predictions)
      
        return wrong_predictions
        
    def plot_misclassified(self,wrong_predictions):
        fig = plt.figure(figsize=(10,12))
        fig.tight_layout()
        mean,std = self.cifar_dataset.calculate_mean_std()
        #mean,std = helper.calculate_mean_std("CIFAR10")
        for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
            img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
            for j in range(img.shape[0]):
                img[j] = (img[j]*std[j])+mean[j]
            
            img = np.transpose(img, (1, 2, 0)) #/ 2 + 0.5
            ax = fig.add_subplot(5, 5, i+1)
            ax.axis('off')
            ax.set_title(f'\nactual : {self.class_names[target.item()]}\npredicted : {self.class_names[pred.item()]}',fontsize=10)  
            ax.imshow(img)  
          
        plt.show()
