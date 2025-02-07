# r_Generator
futhark rune generator from dataset of runes and dataset of carved images. to translate text to image style transfer
sofar:

Here's a bullet-point outline of what the FutharkModel is doing:
## Data Input
  Takes in a batch of rune images (size 3x48x48)
  Normalizes the pixel values to be between 0 and 1
## Convolutional Layers
Conv1:
  Applies 32 filters with a kernel size of 3x3
  Uses padding to maintain the spatial dimensions
  Applies batch normalization to normalize the activations
  Applies ReLU activation function to introduce non-linearity
Conv2:
  Applies 64 filters with a kernel size of 3x3
  Uses padding to maintain the spatial dimensions
  Applies batch normalization to normalize the activations
  Applies ReLU activation function to introduce non-linearity
Conv3:
  Applies 128 filters with a kernel size of 3x3
  Uses padding to maintain the spatial dimensions
  Applies batch normalization to normalize the activations
  Applies ReLU activation function to introduce non-linearity
## Pooling Layers
Max Pooling:
  Reduces the spatial dimensions by half
  Applies after each convolutional layer
## Fully Connected Layers
FC1:
  Takes in the flattened output of the convolutional and pooling layers
  Applies a linear transformation to produce 512-dimensional feature vectors
  Applies ReLU activation function to introduce non-linearity
FC2:
  Takes in the output of FC1
  Applies a linear transformation to produce 256-dimensional feature vectors
  Applies ReLU activation function to introduce non-linearity
FC3:
  Takes in the output of FC2
  Applies a linear transformation to produce 24-dimensional output vectors (one for each rune class)
## Output
  Produces a probability distribution over the 24 rune classes
  Can be used for classification or generation tasks
