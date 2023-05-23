# Image Matching

So these are my ramblings on how to handle new findings in image matching techniques.

There is a need to understand the datasets that has been used to pretrain these different models.
What I've come across. 

# Datasets for Evaluation

Datasets for evaluation are crucial to the task.

# MegaDetph
[link](https://www.cs.cornell.edu/projects/megadepth/)
Mainly used to train single image depth estimation.
Pretty large and comes in huge variations.

# Hpatches
See the repo.

# ScanNet
[link](http://www.scan-net.org/)

# CDVS


# Affine stuff

Test, Train and Implement with our data.

## D2-Net


## LoFTR

[LoFTR](https://github.com/zju3dv/LoFTR) 
The LoFTR repo does not only provide the code to run the LoFTR network but also some examples of improvement, not the SE2-LoFTR which seems really interesting. 
1. QuadTreeAttention(https://github.com/Tangshitao/QuadTreeAttention)


### Article

Uses a dense method, but instead of a cost volume a transformer is used. Cross attention layers.
Self and cross attention layers. (What are these?)

#Detector based -- learn the detector
SuperPoint 
MagicPoint
LIFT


#Detector Free
Contrastive loss.
NCNet
DRCNet


#Result

#### Using the code


## SE2-LoFTR

[se2-loftr](https://github.com/georg-bn/se2-loftr) 

## SIFT-AID

[code](https://rdguez-mariano.github.io/pages/sift-aid.html)

### The article

[Cosine proximity ](https://en.wikipedia.org/wiki/Cosine_similarity)

### The code

The code was fairly easy to use, there is function
that runs sift-aid on two images and then writes two images:
 1. AID\_homography\_matches.png
 2. AID\_panorama.png

The first one is a standard point matching image, drawing lines between matching points.
The second is a transformation of im1 to im2 given the homography from the points.

## HardNet
[HardNet](https://arxiv.org/pdf/1705.10872.pdf)

## Locate

## Affine Net

[AffineNet](https://arxiv.org/pdf/1711.06704.pdf)


## Stereo Matching by training a convolutional neural network to compare image patches

[article](https://arxiv.org/pdf/1510.05970.pdf)

## Deep image homography estimation

[article](https://arxiv.org/pdf/1606.03798.pdf)

## IMAS

[imas](https://github.com/rdguez-mariano/fast_imas_IPOL)

# FlowNet

## Image Matching Benchmark

[git](https://github.com/vcg-uvic/image-matching-benchmark)

## Image Matching Local Features & Beyond

[2020](https://www.youtube.com/watch?v=UQ4uJX7UDB8)
[2021](https://www.youtube.com/watch?v=9cVV9m_b5Ys) 
[2022](https://www.youtube.com/watch?v=Nr-hQG7k1bM) 

# Affine Maps Tests and Experiments

There are several methods to perform image matching between images 
witch are separated by a severe affine transformation.
Or where images that have been collected with different cameras from 
different perspectives.

## Affine Transformations

## Affine invariant feature detectors

## The Code

### Affine2d -- opencv

# ML-based affine invariant features



### Results


# What about detector learning?

### Running CUDA with docker and tensorflow applications

Why are people not making Dockers for their tensorflow applications?
There seems to be a large gap between tf1.x and tf2.x making it hard to build old stuff without addind stuff in other peoples code.

With new nvidia-docker it's easier to set up a deep learning applicaton,
and still use the GPU, through a docker container.

I also got some problems using my later build CUDA11 towards python tf that were built for 
earlier CUDA libraries making it uncompatible.

[Here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) is information on how to use and install docker with nvidia.

[video](https://www.youtube.com/watch?v=jdip_6vTw0s)


To run with dockerized cuda:

'''
  deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
https://stackoverflow.com/questions/70761192/docker-compose-equivalent-of-docker-run-gpu-all-option


