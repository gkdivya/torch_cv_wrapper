# Models

## ResNet
![image](https://user-images.githubusercontent.com/17870236/124294709-9f960e00-db75-11eb-93f7-0cc3825b4c49.png)


## Custom Model 

     Net(
       (conv1): Sequential(
         (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (1): ReLU()
         (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (3): Dropout2d(p=0.01, inplace=False)
         (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (5): ReLU()
         (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (7): Dropout2d(p=0.01, inplace=False)
       )
       (trans1): Sequential(
         (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
         (1): ReLU()
       )
       (conv2): Sequential(
         (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (1): ReLU()
         (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (3): Dropout2d(p=0.01, inplace=False)
         (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
         (5): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
         (6): ReLU()
         (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (8): Dropout2d(p=0.01, inplace=False)
       )
       (trans2): Sequential(
         (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
         (1): ReLU()
       )
       (conv3): Sequential(
         (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
         (1): ReLU()
         (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (3): Dropout2d(p=0.01, inplace=False)
         (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (5): ReLU()
         (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (7): Dropout2d(p=0.01, inplace=False)
       )
       (trans3): Sequential(
         (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(2, 2))
         (1): ReLU()
       )
       (conv4): Sequential(
         (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (1): ReLU()
         (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (3): Dropout2d(p=0.01, inplace=False)
         (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
         (5): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
         (6): ReLU()
         (7): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (8): Dropout2d(p=0.01, inplace=False)
       )
       (gap): Sequential(
         (0): AdaptiveAvgPool2d(output_size=1)
       )
     )
