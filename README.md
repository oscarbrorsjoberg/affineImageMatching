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
Contrastive loss 4d volume.
Earlier work from the author:
1. NCNet
2. DRCNet


#### Method

Standard convolutional architecture with FPN ( Feature Pyramide Network).

## Local Feature CNN
First a CNN to extract two feature maps for each image,  
On a course (1/8) and a fine (1/2) of the image size.


Both the coarse and fine representations go int tohe LoFtr Module.
However there are other opertations, divided into to coarse-level local feature transform and 
coarse to fine module.

The first second step is to take the coarse features and pass then through coarse-level local feature transform.

## 

"Convolutional Neural Networks (CNNs) possess the in-
ductive bias of translation equivariance and locality, which
are well suited for local feature extraction.

-- What?

The downsam-
pling introduced by the CNN also reduces the input length
of the LoFTR module, which is crucial to ensure a manage-
able computation cost."

## The Transformer -- Attention layer

Attention Q, K, V.

Self attention:
Embedded vector from patches in images. From the embedded vector we ectract query vector, key vector and value vector. 
Query - what is important to my value?
Key - what are the other patches responses.
Value - the answer.


## Linear vs Dot-Product (vanilla) Attention
Linear attention is not as computationally complex O(N) instead of O(N2).

Positional encoding in the transformer.
Positional embedding are attached to the transformer?
What are they and why do we need them?
Transformer processes elements of information in parallel.
For each element it combines information from the other elements through self attention.
Each element does this aggregation by itsel, see query from above?
The order of elements in a sequence is unknown to the attention layer, it has to be provided through the positional embedding -- an obvious hint of where the element is located in the sequence.
The embeddings are added to the inital vector representations.
Positional embeddings stay the same independent of the vector representations.

### Back to method

Coarse level matches. After the transformer module, we need to match features. This is done either with optimal trasport layer (SuperGlu) or an dual softmax. See the abrivations on the pretrained models. So the score matrix can be calculated from all the feature vectors. A matching probability is caculated with a double softmax on the two dimensions. The confidence matrix is thresholded and MNN is enforced as a model constraint.

Fine level method. Use the coarse level. Cut out smaller areas. Run them through the LoFTR module. Create the feature maps. Correlate center vector of feature map with all vectors in opposite feature map. 

### The supervision

Learning the stuff comes from two losses, coarse and fine. 
The ground truth labels, which the loss is computed towards, is produced in the same manner as for SuperGlue. These ground truth are compared with the negative log-likelihood towards the confidence matrix.
L2 loss for fine level refinement.

Sinkhorn iterations?

#### Result

HPatches

ScannNet

MegaDepth

Visual Localization Benchmark

#### Using the code

##### Testing
Able to run the notebook in the repo within my docker image.
Testing the different pre-trained worked differently:

1. indoor-ds 
Worked no charm

2. indoor-ot 

3. outdoor-ds 
Worked on the provided data, not on mine.

4. outdoor-ot 

Produced the following error.
  ```

  ```

#### Training

pytorch-lighting -- framework to facilitate training and abelation studies?



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


