
# About this work

This works addresses the problem of automatic image colorization using deep learning techniques. We explore a way of generating colored images from grayscale images with some degree of spatial control, so we could get partial colored images with an artistic effect if desired.


This project was developed as a conclusion work for the course *<a href="http://lvelho.impa.br/ip18/" target="_blank">Fundamentals and Trends in Vision and Image Processing</a>* (August-November, 2018) at IMPA, which had the theme *from Data to Generative Models*.



#### Keynote presentation:

You'll find embedded below the keynote presented 
<iframe src="https://www.icloud.com/keynote/0Wyocnu0kmSktCDVyBD7OOWEQ?embed=true" width="640" height="500" frameborder="0" allowfullscreen="1" referrer="no-referrer"></iframe>

> Presentation may be updated, in relation to that presented during the course, to fix typos, be more coherent with this website and/or complete references and images.

#### Source code

The source code for this project is available at this <a target="_blank">Github repository</a>. The dataset used for training is XXXXXXX.


# Inspirations

There are multiple works about colorization 


Thinking about the problem, there are many things, many results that could be pursued. We can think of automatic photorealistic colorization, color correction, color transfer between images,  colorization or recolonization restricted to some criteria like a specific color palette or specific degrees of some metric like saturation, constrast, brightness etc.


## Artistical neural style transfer

One of the most impressive and deeply used to disseminate the results and advances reached in the area is artistically style transfer. Not only generates beautiful and astonishing images but also It is something of ease access for someone who is a layperson on the subject.

We didn't want to try to extend the results on artistic style transfer, but we did want to achieve something that could be used in an artistic way (fashion).

## Classical works

When we search for image colorization inside a context of machine learning and, more specifically, deep learning we can find a variety of works interested in make a computer guess the colors of a grayscale image making it a photorealistic colored scene. This is a very cool application 

We can find tool which can color images restricted to some hints, like the most prominent color. 

Compute the best colorization given these restrictions or attending these criteria.

Work with histograms seems an interesting proposal due to the fact that a histogram is an statistic measure of the image and we could think about learning some transformations over this metric. 

## Neural networks and colorization


# Goal 

With some knowledge about the problems attacked on the area and the idea of making something with a potential artistic style, we'd like to get results somewhat near to images of partial colorization. That is colorization with a spatial control, so we could achieve an artistic result like that presented in images XXXXXX.

[IMAGES SAMPLES HERE]

# Our approach

After studying many projects about colorization with deep learning, we decided to approach the problem using Generative Adversarial Networks (GANs). We encountered some very good results in other works which didn't use GANs, particularly XXXXX (galera Efros) e XXXXX (harmonization). The former doesn't address the colorization problem specifically, but in the way of trying to harmonize the the style (and the colors) of a piece of image attached to another one, developed some ideias that could be useful. The problem is we didn't figure out an simple way to experiment spatial control in the architectures presented these works. Besides that these works had some sophisticated steps which could need more time to do right than we could provide for the project. 

We based our solution on the code available in [XXXX] (network architecture) and in [XXXX] (training organization). The generator  architecture is composed of XXXXXXXX.

![U-Net architecture of the generator network](https://github.com/hallpaz/Image-Colorization/blob/master/asset/unet.png)

We saw some works where the discriminator network was trained using some famous network as baseline, for example VGG-16. In [XXXXX] they built their own model for the discriminator and we chose to use this model which has an architecture that resembles that of the generator, but only the encondig side, leading to an output of the probability that the image taken from the generator is classified as a fake.

![Discriminator network architecture](http://via.placeholder.com/640x360)
 

To incorporate spatial control in the colorization made using GANs, we tried to follow the ideia of Conditional GANs, changing the network to receive as input not only a grayscale image but also a binary image which would act as a mask. With this approach we expected to teach the network to color only the regions where the pixels had a non-zero value. 

![Generator modified](http://via.placeholder.com/640x360)

As we weren't sure if this approach would prove itself promising, we tested our hypothesis over the XXXX dataset, training XXXX images with a single synthetic mask, an image divided vertically. We expected the network to learn how to generate images half in color and half in grayscale. Unfortunately, this didn't work, as the network began to generate images completely colored, less saturated,  but with a very visible line where it was the hard division of the mask.


## Considerations

change optimization metric (small change)



# Conclusions and next steps



# References

Image style transfer using convolutional neural networks
LA Gatys, AS Ecker, M Bethge
Proceedings of the IEEE Conference on Computer Vision and Pattern
