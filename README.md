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
<br>
**Environment**  
win10 + python 3.6 + pytorch 1.0 + cuda 9.0 + opencv-python 
<br>  
**Usage**  
```
1, take several videos with your mobile phone. Cause I did not rewrite the handle-motion part, so moving objects should not appear in the video!!!
2, put your video in video folder, and create new folder for each video in dataset folder.
3, run split.py, decompose each video into pictures by hand. You may need to comment/uncomment some lines. 
4, run calib.py, calib the phone camera and get intrinsics.
5, write the intrinsics to data_loader.py by hand. Pay attention to the original picture size of your camera, it will affect the scaled intrinsics.
6, run main.py to train.
7, run infer.py to inference.

most params are fixed!
```

Basiclly, I only rewrite the loss calculate part.

All the training data were filmed by my mobile phone.  


