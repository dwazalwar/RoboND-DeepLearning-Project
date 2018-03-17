[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project #

In this project, the main task is to locate and follow a moving target (referred here as Hero) in a simulated drone environment. For this semantic segmentation task, a fully connected deep neural network pipeline has been employed. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques we apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in the industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0]

# Table of Contents

- [Architecture](#architecture)
  * [Fully Convolutional Network](#fully-convolutional-network)
  * [1X1 convolution](#1x1-convolution)
  * [Encoder](#encoder)
  * [Decoder](#decoder)
  * [Batch Normalization](#batch-normalization)
- [Parameter Tuning](#parameter-tuning)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Architecture ##

Semantic segmentation is one of the methods for performing scene understanding, where we assign meaning to each pixel in an image. There can be an argument here why not use a relatively easier approach like object detection methods  (for example SSD or Yolo) and just draw a bounding box around the target. In some frames, it might work but cases where we have a curvy track or cluttered scene with multiple non-hero persons, bounding box approach will not give us accurate enough estimate of Hero's position. Therefore pixel level classification becomes critical in such cases. Here, I have used Fully convolutional Encoder-Decoder architecture seen below in Figure 1. For each incoming image frame from a front-facing camera on the drone, this deep learning pipeline will classify each pixel into 3 classes: Hero, Non-Hero, and Background. Jupyter notebook with step by step execution for Follow Me project can be found here.

[Jupyter Notebook](https://github.com/dwazalwar/RoboND-DeepLearning-Project/blob/master/code/model_training.ipynb)

[image_1]: ./docs/misc/Architecture.png
![alt text][image_1]
### Figure 1. Fully Convolutional Deep Learning Architecture for Follow Me Project ###

### Fully Convolutional Network ###
Convolutional Neural Network layers with its parameter sharing help to learn structures and pixel connectivity information. A fully convolutional network extends this advantage further by keeping spatial information intact throughout the network. A typical FCN usually can be split into two halves: Encoder and Decoder, both separated by 1X1 convolution. Encoder has series of convolution layers and its main objective is to extract features from images. On the other hand, Decoder upscales encoder output so that we have our final segmented output with same spatial dimension as in input image.

### 1X1 convolution ###
 In classic convolutional architectures for any classification task, we have series of convolution layers and in the end, we have few fully connected layers with softmax activation function. Using a fully connected layer in final stages forces us to convert convolution output from 4D to 2D, thus effectively losing pixel location information. Instead, by using 1X1 convolution layer after encoding stage, our output tensor will still be 4D keeping our spatial connectivity information intact.  An added benefit of using 1X1 convolution is now we have the flexibility to work with images of different sizes since convolution operation does not really care about input image size.

 1X1 are also extremely useful as a dimensionality reduction technique. Imagine we have output 'C1' from convolution operation with size of 256X256X200 and we are looking to stack few more convolution layers on top of C1 to extract more features. Before making network deeper, we can instead convolve C1 with set of 50 filters each of size 1X1X200, our output will now have reduced dimension i.e 256X256X50. Such interspersing of 1X1 convolution with classic convolution is common practice to make networks deeper and still keeping similar overall structure. Also apart from dimensionality reduction here 1X1 convolution does introduce new parameters and non-linearity to increase model accuracy.

### Encoder ###
 Encoder has series of Convolution-Relu layers stacked together to extract features from the input image. First layer usually tries to identify basic image characteristics like oriented edges, horizontal/vertical lines or simple blobs of color. The subsequent layers tend to learn more complex features like shapes and eventually objects or combination of objects.

 Usually to extract such multiple features in an image we typically have to increase the depth of classic convolutional layer in our network. While this approach works and does improve our model accuracy, it can become a headache in some cases as more depth means more number of parameters to optimize. Instead, we can use a special technique called separable convolution or sometimes also known as depth-wise separable convolution. The main idea is we perform normal convolution only once over each channel of input, followed by 1X1 convolution to take the previous output and combine them finally to form our output layer. This helps to reduce the number of parameters significantly and improves training speed.

 Here in this architecture, I have added identity block in between similar to building block in ResNet architecture (see Figure 2). This allows the network to learn residual information and preventing degrading network performance when we want more depth.

[image_2]: ./docs/misc/ResnetBuildingBlock.png
![alt text][image_2]
### Figure 2. ResNet Architecture Building Block ###

### Decoder ###
 After encoding stage and 1X1 convolution layers, even though we now have extracted features but still layer dimensions are not same as that of the input image. Therefore at Decoder stage, we need few more operations to get output image with same size as input image.

 One of the important operations in decoding is up-sampling step. Transpose convolution can be used where we just swap forward and backward passes compared to classic convolution. So for example if we have 3X3 input layer, to up sample to 6X6 we simply multiply each pixel of input layer with 6X6 kernel or filter. In tensorflow we handle this using strides = 2 and keeping same padding. For this project, I have used bilinear up-sampling which is relatively faster approach since there are no parameters added by this interpolation technique. Here we perform weighted average of 4 nearest diagonally located neighbor pixels to estimate new pixel value. In comparison to transpose convolution, since we do not have learning layer involved here segmentation accuracy can be slightly less in some cases. However in this project it worked good enough to segment target as well non target person correctly.

 During convolution operation(in encoder), network is trained to focus on specific features and so even though we up-sample this final encoded image to original size, decoder output may still be missing some finer details. To further improve our segmentation accuracy in such cases, we add skip connections at decoder stage to use information from multiple resolutions. Instead of doing element wise addition operation for skip connections, here I have employed layer concatenation. This is easier approach as it allows adding connections without worrying about its depth unlike in case of addition operation.

### Batch Normalization ###
 This is another important technique to speed up the training process. It is a well-proven fact that having normalized inputs help network learn faster. Batch Normalization extends this idea further in doing this not just at the initial input stage, but throughout across network stages. This significantly improves training speed and also allows the use of relatively higher learning rates.

## Parameter Tuning ##
 Selection of architecture is an important step but equally important is how we tune our hyper parameters and then finally select one which gives us the best result.

 Deciding a learning rate for a network means to define your step towards minimizing your loss function. Below is a graph showing different possible types of learning and their impact. A very high learning rate can cause your optimizer to overshoot because of weight changes and also it may not converge. A very low learning rate can affect your training speed and it might take ages to finally come up with an optimized model. For this implementation, I tried learning rates of 0.01, 0.001 & 0.0005 and best results were seen for 0.001.

[image_3]: ./docs/misc/learningrates.jpeg
![alt text][image_3]
### Figure 3. Learning rate types ###

 Figure 4. below shows the training curve for this project and we can clearly see both training as well as validation curve, converge to minimum loss.

[image_4]: ./docs/misc/training_loss_curve.png
![alt text][image_4]
### Figure 4. Final Training curve ###

 Deciding batch size depends on total number of parameters i.e model complexity and also on your GPU specifications. For my architecture batch size of 100 tend to quickly exhaust GPU resources. Keeping the batch size of 64 worked very well in most cases.

 For the number of epochs, I tried 10, 15, 20 and 25. After 12-13 epochs, I saw most of the times training and validation loss reached its optimal value and stabilized enough. Therefore in I decided to keep value to 15. Steps per epoch for training and validation had a relatively lower impact on performance. I tried to lower it to 100, 150 in training and 25 in validation, but model accuracy was affected a little bit. In the end, I decided to keep default values provided in the jupyter notebook.

 After careful optimization trials, these were some of the key hyper-parameters used in final training done on AWS ec2 platform.

 Learning Rate           = 0.001
 Batch Size                = 64
 Number of Epochs   = 15
 Steps per Epoch      = 200
 Validation Steps       = 50

## Results ##
 Evaluation is done on 3 test scenarios:
1. The quad-copter is following the target
2. Images at patrol without target person
3. Images at patrol with a target at a distance

 Below are some images showing results for each of the above scenarios. Left most image is input frame given to the network, the middle is ground truth showing expected output and rightmost is output image from deep learning pipeline.

[image_5]: ./docs/misc/result_image_1.png
![alt text][image_5]
### Figure 5. Quad copter following target : sample output images ###

[image_6]: ./docs/misc/result_image_2.png
![alt text][image_6]
### Figure 6. Quad copter at patrol without target : sample output images ###

[image_7]: ./docs/misc/result_image_3.png
![alt text][image_7]
### Figure 7. Quad copter at patrol with target: sample output images ###

Finally, we use IntersectionOverUnion(IOU) metrics for summarizing model performance. Its description is shown in Figure 8 and also IOU performance for the model can be seen in Figure 9.

[image_8]: ./docs/misc/iou_equation.png
![alt text][image_8]
### Figure 8. IOU Explained ###

[image_9]: ./docs/misc/final_iou.png
![alt text][image_9]
### Figure 9. Overall IOU for Follow Me Project ###

## Future Enhancements ##
 The performance of existing model can be further improved by collecting more data samples for training. For that, we should employ a calculated approach in terms of collecting data from different viewpoints like the target at a far distance, samples with no target and also target in a cluttered environment.

 Deep learning pipeline used in this project is generic and should work well on segmenting any other objects like dog/cat or even cars. Having said that, this network is implemented for identifying people and so each encoder stage is trained to identify features related to that target person. Now if we want to identify completely different object like say dog/cat, we need to learn different set of features which may be unique to that object. Therefore we will have to retrain our input data with new object, allow our encoders to learn those unique object features and finally give us new model which will accurately segment new object.

## References ##
1) Udacity NanoDegree Deep Learning Lesson
2) [Stanford Lecture Series on Convolutional Neural Networks](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
3) [ResNet original paper](https://arxiv.org/pdf/1512.03385.pdf)
4) [Semantic Segmentation Guide](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)
















