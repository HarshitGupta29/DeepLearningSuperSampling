# DeepLearningSuperSampling

The model we use is SRGAN - Super Resolution General Adversary Networks

## Prologue: Why are we doing this ?

This is a really interesting field and we're passionate about computer vision and gaming so we decided to take this project on. ~~theres also a neat prize~~.


## Part I: The Generator Network

Our generator network takes a low resolution image and generates a high resolution image. For this, residual blocks and skip connections were used. The residual block consists of a convolution layer followed by a batch normalization layer, a Leaky ReLU activation, one more convolution layer, and lastly a sum method. We also use Pixel Shuffler in the model to make it more robust.

## Part II: The Discriminator ~~Terminator~~

This network takes a high resolution image produced by the generator network and matches it with high resolution images taken from a camera. It judges if an image is a fake high resolution image or a real high resolution image. In this network we create a stack of layers where a convolution layer followed by a batch norm and a Leaky ReLu. We then take this stack of layers and add them in a line multiple times. In the end we use some Dense and Leaky ReLu, and sigmoid to output a score for the input image.

Basically, generator tries to fool the discriminator and discriminator tries to catch the generator. #TypicalGANBehaviour🕵️🦹

We keep the discriminator constant when the generator is learning and we keep the generator constant when the discriminator is learning. This way both of them learn simultaenously.

## Part III: Efficiency  

Occasionally certain sections of images are quite easy to zoom into like clear water with no waves. To make a high resolution image, one could just copy the the color of a pixel into surrounding pixels. Whereas the same technique cannot be used for a detailed landscape like a grassland. We need very fine details in a high resolution image which we cannot get from just copying a pixel value onto surrounding pixels. To overcome this problem two networks labeled Coarse and Fine are put to use. Firstly, important parts of an image are sampled and assigned weights. After running our coarse network on those parts, the results are judged using the previous weights which include fine details and geometry which our network couldn't obtain. We will choose those specific parts and cut out some more parts near those. Finally the Fine network processes all of these parts. This way our model will be able to understand fine geometries and simple things like water. #MachinesReallyAreStupid

## Part IV: Training

We trained our model using a high resolution dataset in CV, DIVerse 2K. We used Adam Optimizer with a variable learning rate and AutoGrad. #HateComputations



