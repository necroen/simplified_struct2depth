### <p align="center">simplified struct2depth</p>  
This is a simplified(rough) version of struct2depth in pytorch.  
```
handle_motion  False
architecture    simple
joint_encoder  False
depth_normalization  True
compute_minimum_loss  True

train from scratch  
I didn't rewrite the handle motion part code with pytorch !!!
```
![gif](./misc/rst.gif)  
The gif above is the training result of 1634 pictures and 95 epochs. 
<br> 
[**Chinese readme 中文**](./misc/说明.pdf)  
heavily borrow code from [sfmlearner](https://github.com/ClementPinard/SfmLearner-Pytorch) and [monodepth](https://github.com/ClubAI/MonoDepth-PyTorch).  
[original code in tensorflow](https://github.com/tensorflow/models/tree/master/research/struct2depth)  

All the training data were filmed by my mobile phone.  

qq group：369308681