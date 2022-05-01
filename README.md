# DeepLearningSuperSampling

The model we use is SRGAN - Super Resolution General Adversary Networks

## Prologue: Why are we doing this ?

Let's be honest here - ~~we really wanted to dive into this field and since we love computer vision, we took this project on~~ we want money. #BeHonest


## Part I: The Generator Network

Our generator network will take a low resolution image and will try to generate a high resolution image. For this we used residual blocks and skip connections where a residual block consisted of convolution layer followed by a batch normalization layer, a Leaky ReLU activation, one more convolution layer, and lastly with a sum method. Input in a residual block is also fed into sum method, using the skip connection, so that most of the information is not lost. We also use Pixel Shuffler in the end of the model to make it more robust.

## Part II: The Discriminator ~~Terminator~~

This network takes a high resolution image produced by the generator network and matches it with real high resolution images taken from camera. It judges if an image is a fake high resolution image or a real high resolution image. In this network we create a stack of layers where a convolution layer followed by a batch norm and a Leaky ReLu. We then take this stack of layers and add them in a line multiple times. In the end we use some Dense and Leaky ReLu, and sigmoid to output a score for the input image.

Basically, generator tries to fool the discriminator and discriminator tries to catch the generator. #TypicalGANBehaviour🕵️🦹

We keep discriminator constant when generator is learning and we keep generator constant when discriminator is learning. This way both of them learn simultaenously.

## Part III: Efficiency  

Usually some parts of images are really easy to zoom in like clear water with no waves. To make a high resolution image, just copy blue pixels around water so that they blend and you are done. Whereas some parts are really messy like a grass land. We need very fine details in high resolution image which we cannot get from just copy some pixels from here and there. To overcome this problem we train two networks - Coarse and Fine network. We will sample some parts from an image we think are important and will give them some weights. After running our coarse network on it, we will judge using those previous weights, which part actually had fine details and geometry which our network couldn't obtain. We will choose those specific parts and cut out some more parts near those. Then we will run fine network on all of these parts. This way our model will be able to understand fine geometries and simple thinks like water. #MachinesReallyAreStupid

## Part IV: Training

We just train it on most popular high resolution dataset in CV, DIVerse 2K. We use Adam Optimizer with a variable learning rate and AutoGrad. #HateComputations



