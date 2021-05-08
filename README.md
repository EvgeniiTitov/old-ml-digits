## Machine learning

### 5.1 Build a digit recognition script
Use the MNIST dataset, build a script that achieves > 98% accuracy on the test
data


### Reasoning / thoughts
The task at hand is a classification problem. Since the problem definition does
not mention whether I am allowed to use any pretrained SOTA networks, I will
assume it is okay and will start with it. I will be using *PyTorch* to complete
the task.

**train.py** - model training. If you'd like to test it, please specify the training
conditions in the config.py. The script will work with some of the SOTA models 
such as ResNet, VGG, AlexNet etc.

To score the model please run **inference.py** with the validation_folder arg. The 
trained weights and other artifacts could be found under *output*



---
#####Training process (for myself):

The problem at hand is super simple, so using a heavy large SOTA network here 
is definitely an overkill. And yet, it is the fastest way to get the requested
result. A simple network with a couple of convolutional layers should to the 
trick as well without any issues.

Run 1:
Started with a pretrained ResNet18. No augmentation, all conv layers frozen, 
training only the classifier at the end. Standard LR, 10 batch size, 8 epoch. 
The val accuracy stalled at ~0.968

Run 2:
Since the task is simple, I am hesitant to unfreeze anything, it feels like I could
improve the performance by tuning other hyper parameters. Tried to apply some
augmentation - rotations, and larger batch size of 25. The model was training
super bad - I guess the batch size is probably too large: going down the loss
function's slope gets problematic with the large batch size as we get *pulled* 
towards multiple local minimums simulataneously resulting in poor training.

Run 3:
Decreated batch size to 13, kept augmentation. It didnt work as augmentation
usually helps to avoid overfitting and not to achieve better training results.

Run 4: Unfroze the conv layers - training all layers now.
Requested accuracy achieved on the second epoch. I wonder if I could reach it
without training the conv layers.
On the epoch 7 the best accuracy 0.9966 was reached. Potentially, could have got
even better results if I'd trained for longer.