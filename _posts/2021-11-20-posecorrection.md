---
layout: post
title: Human Pose Correction
subtitle: Object Detection using NVIDIA Jetson Nano
tags: [Object Detection, Machine Learning]
---

This is how we always end up on a chair...
![Photo](/assets/img/pc1.png)

## Motivation 

Since the pandemic, we are stuck at home and spend so many hours infront of a desk. What happens when we're sitting on a chair? We start to lean forward, lean to the side, cross out legs and rest on our chin. To save us from intense back pain, I came up with the idea to provide a program that provide body posture correction. 

## What you to make a Pose Correction program

You need a camera that can be connected to the Jetson Nano developer tool, image dataset, and a neural network.

I used the Raspberry camera (because it was really affordable) and crawled human pose images on Google. I first intended to use the Coco dataset, however, the dataset size (20G) exceeded Jetson Nano's memory size so I couldn't use it.
To improve the accuracy and lower the latency, instead of using the conventional Resnet, Densenet, and MobileNet networks, I chose Mnasnet.

## The Pose Correction algorithms

For pose detection, I used the trt pose code from [here](https://github.com/NVIDIA-AI-IOT/trt_pose).

After you trained your neural network, you need an algorithm that decides whether the person is in a good posture or not.
This is what I used for my project. I used the x/y coordinates of the body parts.

~~~
def check_slump(neck_y,nose_y):
    if neck_y != 0 and nose_y != 0 and nose_y > neck_y :
        return False
    else :
        return True

def check_tilted_left(leftshoulder_x, leftear_x):
    if leftshoulder_x != 0 and leftear_x != 0 and leftshoulder_x < leftear_x :
        return False
    else :
        return True

def check_tilted_right(rightshoulder_x, rightear_x):
    if rightshoulder_x != 0 and rightear_x != 0 and rightshoulder_x > rightear_x :
        return False
    else :
        return True

def check_tilted_pelvis(leftpelvis_y, rightpelvis_y):
    if leftpelvis_y != 0 and rightpelvis_y != 0 and not -0.02 <=(leftpelvis_y - rightpelvis_y)<=0.02 : 
        return False
    else :
        return True

def check_knee(leftknee_x, rightknee_x) :
    if leftknee_x != 0 and rightknee_x != 0 and not -0.21 <= (leftknee_x - rightknee_x)<=0.21 :
        return False
    else :
        return True

def check_ankle(leftknee_x, leftankle_x) :
    if leftknee_x != 0 and leftankle_x != 0 and  leftknee_x + 0.02< leftankle_x :
        return False
    else :
        return True

def check_headdrop(lefteye_y, leftear_y) :
    if lefteye_y != -1 and leftear_y != -1 and lefteye_y > leftear_y :
        return False
    else :
        return True
~~~

## Demo Video

Here is a gif demo showing how the program works!
![gif](/assets/img/pc_SparkVideo.gif)





