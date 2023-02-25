### Fruit Classification Project

1. Dataset Creation
- Images of Apple, Guava, Orange, and Mandarine are from Fruit 360. https://www.kaggle.com/datasets/moltean/fruits
- To transfrom the color of the fruit. First, I extract the object from the white background image to get the mask of the object.
- Then get the HSV format of the image.
- Adjust the HUE value that I can transform the color to green, red or yellow.
- Write the images to my local disk
- The structure of the data is \
-> SyntheticData/ Training_Data \
---- Apple_Green\
---- Apple_Red \
---- Apple_Yellow \
---- Guava_Green \
---- Guave_Red \
---- .....  \

- I also create a test dataset.

2. Model training
- I use CNN model to classify 12 classes.
- Also implement MLFlow to track the data.
- Based on the graph of training and validation loss, the model performs quite well in this dataset.

3. Multi-Label vs Multi-Classification
- In multi-classification, all the classes are mutually exclusive. For multi-labels, an instances can belong to a multiple classes.
- For this dataset, this can be a multi-label classfication problem. Because an on can be different type of fruit(apple, guava, orange, mandarine) and, at the same time, can be either (red, green, yellow). 