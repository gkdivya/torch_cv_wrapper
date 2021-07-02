# torch-cv-wrapper
Torch based library for developing computer vision models 

## Features
- Following model implementations on PyTorch for CIFAR dataset
    * ResNet18
    * ResNet34
    * CustomModel

- **Albumentations** for Image augmentations 
- **Gradcam** for explaining the model output
- **Util functions** for training, testing, plotting metrics, Tensorboard integration

## Folder Structure of the library

    |── config
    |   ├── config.yaml    
    ├── dataloader  
    |   ├── albumentation.py 
    |   ├── load_data.py
    ├── model  
    |   ├── custommodel.py 
    |   ├── resnet.py
    ├── utils  
    |   ├── __init__.py 
    |   ├── train.py 
    |   ├── test.py 
    |   ├── plot_metrics.py 
    |   ├── helper.py 
    |   ├── gradcam.py 
    ├── main.py     
    ├── README.md  

## References
[Kuangliu's PyTorch on CIFAR repo](https://github.com/kuangliu/pytorch-cifar)
[Tensorboard](https://www.youtube.com/watch?v=pSexXMdruFM&ab_channel=deeplizard)
[GradCam with PyTorch](https://github.com/kazuto1011/grad-cam-pytorch)
