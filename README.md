# Visualise Hidden Layers on Custom Data using Transfer Learning Architectures

This is summary readme for Visualising hidden layers on custom data using Transfer Learning Architectures such as ResNet, Inception ResNetV2, InceptionV3 by running [hidden_layers_visualisation](hidden_layers_visualisation.ipynb) code using Colab

1. Download dataset, i.e., Cats and Dogs consisting 25,0000 images, divided into `train\` and `test\` folders having structure:

```
train\
	cats\
		cat1.jpg
		cat2.jpg
		.....
	dogs\
		dog1.jpg
		dog2.jpg
		.....
test\
	1.jpg
	2.jpg
	......
```

2. Using different Transfer Learning architectures like ResNet, InceptionV3, MobileNet, VGG by tweaking and importing necessary packages Keras functions such as:

# ResNet50:
```
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
```
# VGG16/VGG19:
```
from keras.applications.vgg16 import VGG16                 #from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input      #from keras.applications.vgg19 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)       #base_model = VGG19(weights='imagenet')
``` 
# InceptionV3:
```
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
```
# MobileNet:
```
from keras.applications.mobilenet.MobileNet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

model = MobileNet(weights='imagenet')
```



---
## License & Copyright

@ Ashish & Team

***
