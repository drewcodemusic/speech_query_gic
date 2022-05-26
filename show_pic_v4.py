
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 讀訓練好的model

def query_picture(speech_result):

    with open('model_trained.json','r') as f:
        model_json = json.load(f)
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model_trained.h5')

    test_path = 'data/apples-vs-bananas/test'
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=test_path,target_size=(224,224),classes=['apple','banana'],batch_size=10)

    test_imgs, test_labels = next(test_batches)
    # plotImages(test_imgs)
    print(test_labels)

    test_batches.classes

    predictions = loaded_model.predict(x=test_batches,verbose=0)
    predict_results = np.round(predictions)

    # 0.0.3b ADD
    # request_data = input('Input your request data (apple/banana):')

    data_order = 0
    for result in predict_results:
        print(result)
        if speech_result == 0:
            if np.array_equal(result,[1, 0]):
                plt.imshow(test_imgs[data_order])
                plt.show()
                break
        else:
            if np.array_equal(result,[0, 1]):
                plt.imshow(test_imgs[data_order])
                plt.show()
                break
        data_order += 1