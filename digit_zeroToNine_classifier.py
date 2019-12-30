from scipy.misc import imread, imresize
import numpy as np
import re
import base64
import tensorflow as tf


class digitZeroToNineClassifier():
    def __init__(self):
        self.model = None
        self.output_image = 'path/to/save/image/output.png'
        self.image_data = 'path/of/digit/image/digit.png'
        self.folder_name = 'path/to/save/model/'

    def model_initialisation(self):
        json_file = self.folder_name + 'digit_model.json'
        with open(json_file, 'r') as f:
            model_json = f.read()

        self.model = tf.keras.models.model_from_json(model_json)
        # load weights into new model
        model_file = self.folder_name + 'digit_model.h5'
        self.model.load_weights(model_file)

    def convertImage(self):
        imgstr = re.search(r'base64,(.*)', str(self.image_data)).group(1)
        with open(self.output_image, 'wb') as output:
            output.write(base64.b64decode(imgstr))

    def prediction(self):
        self.model_initialisation()
        self.convertImage()
        x = imread(self.output_image, mode='L')
        # make it the right size
        x = imresize(x, (28, 28))
        # convert to a 4D tensor to feed into our model
        x = x.reshape(1, 28, 28, 1)
        # perform the prediction
        out = self.model.predict(x)
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])

digit_class_var = digitZeroToNineClassifier()
digit_class_var.prediction()
