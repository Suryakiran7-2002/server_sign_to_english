import socket               # Import socket module
from pickle import dumps,loads
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('modelfinal_100.h5',compile = False)
model.compile()
actions = np.array(['angry',
 'apple',
 'bad',
 'come',
 'cook',
 'dad',
 'dance',
 'drink',
 'eat',
 'fan',
 'friend',
 'go',
 'good',
 'happy',
 'he',
 'hello',
 'help',
 'home',
 'hospital',
 'how',
 'hungry',
 'like',
 'look',
 'me',
 'meet',
 'mom',
 'money',
 'nice',
 'no',
 'play',
 'read',
 'run',
 'sad',
 'school',
 'sleep',
 'smell',
 'sorry',
 'thanks',
 'that',
 'today',
 'tomorrow',
 'ugly',
 'umbrella',
 'want',
 'what',
 'when',
 'which',
 'yes',
 'yesterday',
 'you'])
print('model complied')


s = socket.socket()         
host = '192.168.1.18' 
port = 12345          
s.bind((host, port))  
s.listen(1)
print('listening')

while True:
    sequence = []
    sentence = []
    threshold = 0.8
    c,addr = s.accept()
    print ('Got connection from', addr )
    while True:

        try:
            x = c.recv(13447)
            keypoints = loads(x)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0),verbose=None)[0]
                    pred = actions[np.argmax(res)]
                    print(pred)
        except:
            break
    c.close()
    print('listening, again')
    

    