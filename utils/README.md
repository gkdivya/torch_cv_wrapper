[Albumentation](#albumentation)

# albumentation.py


# Class Activation Map -  GradCAM

Gradient-weighted Class Activation Mapping (GradCAM) uses the gradients of any target concept (say logits for 'dog' or even a caption), 
flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. _


Steps:
1. Load an image that is processed by this model 
2. Infer the image and get the topmost class index
3. Take the output of the final convolutional layer
4. Compute the gradient of the class output value w.r.t to L feature maps
     Feature map would be related/ non related/ inversely proportional to the class identified
5. Pool the gradients over all the axes leaving out the channel dimension
6. Weigh the output feature map with the computed gradients (+ve)
7. Average the weighted feature maps along channels
8. Normalize the heat map to make the values between 0 and 1

![image](https://user-images.githubusercontent.com/17870236/124298252-ad4d9280-db79-11eb-9986-7e3fb9801a1d.png)

## Code 
![image](https://user-images.githubusercontent.com/17870236/124299234-d15da380-db7a-11eb-8dac-bf4213c475d3.png)

Important points to note:
- For gradient map to be generated, last output layer at least should be of 8x8/7x7 size.
- In an object detection use case, Gradcam highlights the object which is more highlighted/object with higher amplitude values not just the object of big size 

## Sample GradCAM
![image](https://user-images.githubusercontent.com/17870236/124299966-a9227480-db7b-11eb-8805-3558a4b309f5.png)

# albumentation
# Image Augmentation
Albumentation Library is being used to apply image augmentation

- Faster than TorchVision inbuilt augmentation
- Better support for segmentation and object detection dataset with "label preserving transformations"

Visualization of Image Augmentation on CIFAR-10 samples
![image](https://user-images.githubusercontent.com/17870236/124300138-d96a1300-db7b-11eb-8766-a70505c61b94.png)
