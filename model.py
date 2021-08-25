# environment set up
            # create tf conda env via command: 
            # conda create -n tf tensorflow
            # conda activate tf
            # pip install pandas
            # conda install sentencepiece--need this to import tokenization downstream
            # select the tf interpreter

import requests
req  = requests.get('https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')
open('tokenization.py' , 'wb').write(req.content)


import tokenization
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tokenization
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import functions

root_path = os.getcwd()

df = pd.read_csv(os.path.join(root_path, 'Reviews.csv'))

df.columns


label = preprocessing.LabelEncoder()
y = label.fit_transform(df['Score'])
y = to_categorical(y)
print(y[:5])

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

max_len = 512

data_input = functions.bert_encode(df.Text, tokenizer)

model = functions.build_model(bert_layer, max_len)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    data_input, y,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1
)





