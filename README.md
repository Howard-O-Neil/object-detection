# object-detection
## Description
I tried to replicate the R-CNN approach in the original paper but deploy a few changes
+ Make use of pre-trained VGG-16 on imagenet
+ Keep the softmax layer instead of using SVM
+ Keep 1000 output label instead of (1000 + 1, with an additional neuron for detecting background)
+ Deploy regression model, which implement deep-learning bounding box regression (5 dense layers) with no activation function
+ Calculate regression loss by calculating SSE of each prediction by 4 output neurons (dx, dy, dw, dh). The result will be the mean value of 4 SSEs

## Result
![image](https://user-images.githubusercontent.com/64292857/153789129-5a01c4c7-63cc-4e0c-8480-5f8819db77a1.png)
![image](https://user-images.githubusercontent.com/64292857/153789143-916e0181-ee50-42d4-bfc4-8521bc275553.png)
![image](https://user-images.githubusercontent.com/64292857/153789492-fa40743f-8fa1-48d5-8763-21f085313a15.png)

The above example is pretty easy, but this result is also easier to inspect than all others which i have experimented.

Because we only predict by 1000 labels (object label only), the model doesn't seem to be able to filter the real ground-truth object with the background stuffs.

I'll keep develop this in another branch, where i'll deploy MobileNet pretrained model instead of VGG-16, with additional model for detecting background/object.
