# About this work

This works addresses the problem of automatic image colorization using deep learning techniques. We explore a way of generating colored images from grayscale images with some degree of spatial control, so we could get partial colored images with an artistic effect if desired.


This project was developed as a conclusion work for the course *<a href="http://lvelho.impa.br/ip18/" target="_blank">Fundamentals and Trends in Vision and Image Processing</a>* (August-November, 2018) at IMPA, which had the theme *from Data to Generative Models*.



#### Keynote presentation:

You'll find embedded below the keynote presented on the last class of the course.

<iframe src="https://www.icloud.com/keynote/0Wyocnu0kmSktCDVyBD7OOWEQ?embed=true" width="640" height="500" frameborder="0" allowfullscreen="1" referrer="no-referrer"></iframe>

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

# Goal 

With some knowledge about the problems attacked on the area and the idea of making something with a potential artistic style, we'd like to get results somewhat near to images of partial colorization. That is colorization with a spatial control, so we could achieve an artistic result like that presented in images XXXXXX.

[IMAGES SAMPLES HERE]

# Our approach

- 

Optimizations
- Com tantas opções, o que escolher?
- Life is a NP Complete problem
- We also can Feed forward + back propagation (sometimes)

- Interessante
- Explorar os conhecimentos apresentados no curso
- Adição ao que já foi feito
- Factível dentro do tempo que resta

Estratégias Gerais

- Regressão
- Classificação
- GANs


Estratégia 1
1. Aproveitar um trabalho de transferência de estilo e modificar a função de perda de estilo para tentar preservar silhuetas e modificar cor
2. Explorar histograma - non spatial information
3. Histogram Matching
4. Efeito amarronzado, sépia (brown-is) L2 metric over color space

Estratégia 2:

# References

Image style transfer using convolutional neural networks
LA Gatys, AS Ecker, M Bethge
Proceedings of the IEEE Conference on Computer Vision and Pattern

