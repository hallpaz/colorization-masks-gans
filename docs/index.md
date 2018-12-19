# About this work

This works addresses the problem of automatic image colorization using deep learning techniques. We explore a way of generating colored images from grayscale images with some degree of spatial control, so we could get partial colored images with an artistic effect if desired.


This project was developed as a conclusion work for the course *<a href="http://lvelho.impa.br/ip18/" target="_blank">Fundamentals and Trends in Vision and Image Processing</a>* (August-November, 2018) at IMPA, which had the theme *from Data to Generative Models*.


# Inspirations

There are multiple works about colorization . Thinking about the problem, there are many things, many results that could be pursued. We can think of automatic photorealistic colorization, color correction, color transfer between images,  colorization or recolonization restricted to some criteria like a specific color palette or specific degrees of some metric like saturation, constrast, brightness etc. In this section we'll present some ideas and related works that were used as technical references at this work or inspired us in some way.


## Artistic neural style transfer

One of the most impressive and deeply used  to disseminate the results and advances reached in the area is artistically style transfer. Not only it generates beautiful and astonishing images but also It is something of ease access for someone who is a layperson on the subject. When we show people three images like those in Figure 1, they can easily understand the relationship between them and usually ask the question: *"did a computer do that?"*.


![Gatys, 2016](https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/style_transfer.png?raw=true)


The work of **GATYS** addresses artistic style transfer using neural networks as a patterns recognizer and an optimization tool, presenting the features that should be computed and used from the layers of the neural network and also how to compute an objetive function with them. We didn't want to try to extend the results on artistic style transfer, but we did want to achieve something that could be used in an artistic way. This application could be either by generating new images directly or by acting as a tool to facilitate artist's work.

## Classical works

Following the ideia of style transfer, one could think about a "color transfer" or "recoloring tool", which could be thought as a a way of transposing the colors of a source image or a selected palette to a target image. Many years ago, Reinhard et Al [NUMBER] presented a color transfer algorithm based on statistical metrics like the **mean** and **standard deviation** of the values in each channel of the image — looking at the LAB space. 

In a first moment this approach seemed particularly interesting for us, because machine learning deals heavily with statistics. Trying to figure out a way of training a neural network to manipulate this metrics and generate images with a desired property could be an interesting work. However, once we have metrics like mean, standard deviation or even the histogram of colors of an image, we lose the correlation with spatial information and it's not clear how we could deal with it in the context of neural networks.

![Reinhard, 2002](https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/color_transfer.png?raw=true)


On the other hand, acting with spatial information, Levin [NUMBER] presented a way of coloring a grayscale a image using some inputs of the user as hints. They present a closed form solution to an optimization problem based on the hypothesis that nearby pixels with similar luminance values should have similar chrominance values. The ideia behind this paper could help building an interesting tool for assisted colorization.

![Levin, ANO-NUMBER](https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/levin_colorization_optimization.png?raw=true)

Note that none of these works deals with neural networks, but since they are based on optimization techniques and statistical metrics of the images, it is reasonable to suppose that it's possible to construct and train neural networks capable of reproducing similar results. We didn't follow this path though.


## Neural networks and colorization

When we search for image colorization in the context of deep learning we can find a variety of works interested in make a computer guess the colors of a grayscale image making it a photorealistic colored scene. Based only on the value of luminance and some spatial information, we have largely degrees of freedom in the task of computing color information, which makes the problem ill-posed.

TALK ABOUT SOME WORKS OF COLORIZATION

# Goal 

With the idea of making something with a potential artistic style and some knowledge about the solutions which addressed some problems in the colorization area, we decided to make colorization with some degree of spatial control. We would like to build a way to get partial colorization of images and achieve artistic results like those presented in figures [NUMBER].


![Images with partial colorization](http://via.placeholder.com/640x360)

# Our approach

After studying many projects about colorization with deep learning, we decided to approach the problem using Generative Adversarial Networks (GANs). We found some very good results in other works which didn't use GANs, particularly [NUMBER] (galera Efros) and NUMBER (harmonization). The former doesn't address the colorization problem specifically, but the ideas developed to harmonize the style and colors of images pasted into a painting could be useful. The problem is we didn't figure out a simple way to experiment spatial control in the architectures of networks presented in these works. Besides that, these works had some delicate steps which could need more time to do correctly than we could provide for the project. 

### The network
We based our solution on the code available in **[NUMBER]** (network architecture) and in **[NUMBER]** (training organization). The generator  architecture is composed of **LAYERS.**

![U-Net architecture of the generator network - NUMBER](https://github.com/hallpaz/Image-Colorization/blob/master/asset/unet.png?raw=true)

We read some works where the discriminator network was trained using some pre-trained network as baseline, for example VGG-16. In [NUMBER] they built their own model for the discriminator and we chose to use this model which has an architecture that resembles that of the generator, but only the encondig side, leading to an output of the probability that the image taken from the generator is classified as a fake.

![Discriminator network architecture](http://via.placeholder.com/640x360)
 
### The masks

To incorporate spatial control in the colorization made using GANs, we tried to follow the ideia of Conditional GANs, changing the generator network to receive as input not only a grayscale image but also a binary image which would act as a mask. The discriminator network was also modified to receive the same mask in addition to the colored image. In both cases the mask ended up concatenated as another channel of the other image. The last change we had to do was to add a term to the total loss function penalizing the distance between the generated colors and the desired colors with the squared L2 norm. With this approach we expected to teach the network to color only the regions where the pixels had a non-zero value. 

As we weren't sure if this approach would prove itself promising, we tested our hypothesis over the 17 Categories Flower Dataset, training XXXX images with a single synthetic mask, an image divided vertically. We expected the network to learn how to generate images half in color and half in grayscale. Unfortunately, this didn't work, as the network began to generate images completely colored, less saturated,  but with a very visible line where it was the hard division of the mask.

![Generator modified](http://via.placeholder.com/640x360)

Before training the modified network with masks, we trained the original networks using only the images from our selected dataset to check if it was enough to make the network learn how to color the flowers. The colorization obtained was "fine", with a bias to the yellow color. The network identified the flowers and usually painted them with a yellow color, but the leaves, the sky and other elements were painted reasonably.

![Generator modified](http://via.placeholder.com/640x360)


Although the results of this experiment weren't encouraging, we decided to try again using a set of masks, instead of a single one. We made 10 synthetic masks using stripes and grids in a graphic editor software and trained the network again (starting from random weights). This time we got results near to that we were expecting, as we show in figures **NUMBERS**.

![Generated with half vertical mask](https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/half_vertical_flower.png?raw=true)


![Generator modified](http://via.placeholder.com/640x360)


![Generator modified](http://via.placeholder.com/640x360)

# Considerations

We trained the GANs for 80 epochs combining each image with all masks during the training. We used *Google's Colaboratory* environment to train the networks powered by a GPU with support to *Nvidia's CUDA*.

Looking to the results shown in the figures [NUMBERS], we can see that the generated images were painted accordingly to the synthetic masks used. 

change optimization metric (small change)



# Next Steps

The results achieved with this project suggests us that research on the use of GANs conditioned to binary masks is a promising direction. Due to the nature of these first experiments, we can't draw a definitive conclusion yet. We conducted expriments with synthetic masks that were unrelated to the contents of the images — the masks had very specific patterns. We think it's necessary to train the network using a wide variety of masks: synthetic masks unrelated to the images, segmentation masks of the contents of the images, noisy masks and random masks to give some examples.

After these experiments we'll have enough information to draw robust conclusions about this proposal. It's important to have in mind that it might be necessary to also change the networks architectures to increase its capacity. Another significant change we could make is to train the networks with a bigger and more diverse dataset. At first, we think it's possible to proceed with the [112 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

having success with the previous steps and thinking about a tool for artist's to use, we could integrate an image segmentation module (neural network) to the pipeline of the network and allow the user to control which objects or which parts of an object should be colored.

# References

#### TODO: FIX ALL REFERENCES

1. Image style transfer using convolutional neural networks
LA Gatys, AS Ecker, M Bethge, Proceedings of the IEEE Conference on Computer Vision and Pattern
2. [Erik Reinhard , Michael Ashikhmin , Bruce Gooch , Peter Shirley, Color Transfer between Images, IEEE Computer Graphics and Applications, v.21 n.5, p.34-41, September 2001](https://dl.acm.org/citation.cfm?id=618848) [doi>[10.1109/38.946629](https://dx.doi.org/10.1109/38.946629)]
3. Nilsback, M-E. and Zisserman, A.  A Visual Vocabulary for Flower Classification. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2006) http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback06 {pdf,ps.gz}
4. Nilsback, M-E. and Zisserman, A. Automated flower classification over a large number of classes. Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008) http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.
5. Zhang, Richard and Isola, Phillip and Efros, Alexei A . Colorful Image Colorization. ECCV  2016
6. Nazeri, Kamyar and Ng, Eric and Ebrahimi, Mehran. Image Colorization Using Generative Adversarial Networks International Conference on Articulated Motion and Deformable Objects  p.  85--94. 2018.
7. 

# Other materials

#### Keynote presentation:

You'll find embedded below the keynote presented on the last class of the course.
<iframe src="https://www.icloud.com/keynote/0Wyocnu0kmSktCDVyBD7OOWEQ?embed=true" width="640" height="500" frameborder="0" allowfullscreen="1" referrer="no-referrer"></iframe>

> Presentation may be updated, in relation to that presented during the course, to fix typos, be more coherent with this website and/or complete references and images.

#### Source code

The source code for this project is available at this <a href="https://github.com/hallpaz/colorization-masks-gans" target="_blank">Github repository</a>. The dataset used for training  is the [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) **[NUMBER]**.
