# About this work

This works addresses the problem of automatic image colorization using deep learning techniques. We explore a way of generating colored images from grayscale images with some degree of spatial control, so we could get partial colored images with an artistic effect if desired.


This project was developed as a conclusion work for the course *<a href="http://lvelho.impa.br/ip18/" target="_blank">Fundamentals and Trends in Vision and Image Processing</a>* (August-November, 2018) at IMPA, which had the theme *from Data to Generative Models*.



#### Keynote presentation:

You'll find embedded below the keynote presented on the last class of the course.
<iframe src="https://www.icloud.com/keynote/0Wyocnu0kmSktCDVyBD7OOWEQ?embed=true" width="640" height="500" frameborder="0" allowfullscreen="1" referrer="no-referrer"></iframe>

> Presentation may be updated, in relation to that presented during the course, to fix typos, be more coherent with this website and/or complete references and images.

#### Source code

The source code for this project is available at this <a href="https://github.com/hallpaz/colorization-masks-gans" target="_blank">Github repository</a>. The dataset used for training  is the [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) **[XXX]**.


# Inspirations

There are multiple works about colorization . Thinking about the problem, there are many things, many results that could be pursued. We can think of automatic photorealistic colorization, color correction, color transfer between images,  colorization or recolonization restricted to some criteria like a specific color palette or specific degrees of some metric like saturation, constrast, brightness etc. In this section we'll present some ideas and related works that were used as technical references at this work or inspired us in some way.


## Artistical neural style transfer

One of the most impressive and deeply used  to disseminate the results and advances reached in the area is artistically style transfer. Not only it generates beautiful and astonishing images but also It is something of ease access for someone who is a layperson on the subject. When we show people three images like those in Figure 1, they can easily understand the relationship between them and usually ask the question: *"did a computer do that?"*.


![Discriminator network architecture](http://via.placeholder.com/640x360)


The work of **GATTYS** addresses artistic style transfer using neural networks as a patterns recognizer and an optimization tool, presenting the features that should be computed and used from the layers of the neural network and also how to compute an objetive function with them. We didn't want to try to extend the results on artistic style transfer, but we did want to achieve something that could be used in an artistic way. This application could be either by generating new images directly or by acting as a tool to facilitate artist's work.

## Classical works

Following the ideia of style transfer, one could think about a "color transfer" or "recoloring tool", which could be thought as a a way of transposing the colors of a source image or a selected palette to a target image. Many years ago, Reinhard et Al [XXXX] presented a color transfer algorithm based on statistical metrics like the **mean** and **standard deviation** of the values in each channel of the image — looking at the L*a*b space. 

In a first moment this approach seemed particularly interesting for us, because machine learning deals heavily with statistics. Trying to figure out a way of training a neural network to manipulate this metrics and generate images with a desired property could be an interesting work. However, once we have metrics like mean, standard deviation or even the histogram of colors of an image, we lose the correlation with spatial information and it's not clear how we could deal with it in the context of neural networks.

On the other hand, acting with spatial information, Levin [XXXXX] presented a way of coloring a grayscale a image using some inputs of the user as hints. They present a closed form solution to an optimization problem based on the hypothesis that nearby pixels with similar luminance values should have similar chrominance values. The ideia behind this paper could help building an interesting tool for assisted colorization.

Note that none of these works deals with neural networks, but since they are based on optimization techniques and statistical metrics of the images, it is reasonable to suppose that it's possible to construct and train neural networks capable of reproducing similar results. We didn't follow this path though.


## Neural networks and colorization

When we search for image colorization inside a context of machine learning and, more specifically, deep learning we can find a variety of works interested in make a computer guess the colors of a grayscale image making it a photorealistic colored scene. This is a very cool application...


# Goal 

With some knowledge about the problems attacked on the area and the idea of making something with a potential artistic style, we'd like to get results somewhat near to images of partial colorization. That is colorization with a spatial control, so we could achieve an artistic result like that presented in images XXXXXX.

[IMAGES SAMPLES HERE]

# Our approach

After studying many projects about colorization with deep learning, we decided to approach the problem using Generative Adversarial Networks (GANs). We found some very good results in other works which didn't use GANs, particularly XXXXX (galera Efros) e XXXXX (harmonization). The former doesn't address the colorization problem specifically, but in the way of trying to harmonize the the style (and the colors) of a piece of image attached to another one, developed some ideias that could be useful. The problem is we didn't figure out an simple way to experiment spatial control in the architectures presented these works. Besides that these works had some sophisticated steps which could need more time to do right than we could provide for the project. 

We based our solution on the code available in **[XXXX]** (network architecture) and in **[XXXX]** (training organization). The generator  architecture is composed of **XXXXXXXX.**

![U-Net architecture of the generator network](https://github.com/hallpaz/Image-Colorization/blob/master/asset/unet.png)

We saw some works where the discriminator network was trained using some famous network as baseline, for example VGG-16. In [XXXXX] they built their own model for the discriminator and we chose to use this model which has an architecture that resembles that of the generator, but only the encondig side, leading to an output of the probability that the image taken from the generator is classified as a fake.

![Discriminator network architecture](http://via.placeholder.com/640x360)
 

To incorporate spatial control in the colorization made using GANs, we tried to follow the ideia of Conditional GANs, changing the network to receive as input not only a grayscale image but also a binary image which would act as a mask. With this approach we expected to teach the network to color only the regions where the pixels had a non-zero value. 

![Generator modified](http://via.placeholder.com/640x360)

Before training the modified network with masks, we trained it as we got it using only the images from [XXX] to check if it was enough to make the network learn how to color the flowers. The colorization obtained was "Ok", with a bias to the yellow color. The network identified the flowers and usually painted them with a yellow color, but the leaves, the sky and other elements were painted reasonably. 

As we weren't sure if this approach would prove itself promising, we tested our hypothesis over the 17 Categories Flower Dataset, training XXXX images with a single synthetic mask, an image divided vertically. We expected the network to learn how to generate images half in color and half in grayscale. Unfortunately, this didn't work, as the network began to generate images completely colored, less saturated,  but with a very visible line where it was the hard division of the mask.


![Generator modified](http://via.placeholder.com/640x360)


## Considerations

change optimization metric (small change)



# Conclusions and next steps



# References

#### TODO: FIX ALL REFERENCES

1. Image style transfer using convolutional neural networks
LA Gatys, AS Ecker, M Bethge, Proceedings of the IEEE Conference on Computer Vision and Pattern
2. [Erik Reinhard , Michael Ashikhmin , Bruce Gooch , Peter Shirley, Color Transfer between Images, IEEE Computer Graphics and Applications, v.21 n.5, p.34-41, September 2001](https://dl.acm.org/citation.cfm?id=618848) [doi>[10.1109/38.946629](https://dx.doi.org/10.1109/38.946629)]
3. Nilsback, M-E. and Zisserman, A.  A Visual Vocabulary for Flower Classification.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2006) http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback06.{pdf,ps.gz}
