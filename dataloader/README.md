# Files

[albumentation.py](#albumentation) </br>
[load_data.py](#load_data) </br>

# albumentation

Albumentation Library is being used to apply image augmentation

- Faster than TorchVision inbuilt augmentation
- Better support for segmentation and object detection dataset with "label preserving transformations"

Visualization of Image Augmentation on CIFAR-10 samples

![image](https://user-images.githubusercontent.com/17870236/124309975-2b656580-db89-11eb-840b-ca361c459a69.png)

# load_data

Dataset can all be passed to a torch.utils.data.DataLoader which can load multiple samples in parallel using torch.multiprocessing workers. 
