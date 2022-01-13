#%%
import pickle

with open('kana_chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)
print(chars_list)

def is_invalid(message):
    is_invalid =False
    for char in message:
        if char not in chars_list:
            is_invalid = True
    return is_invalid
    return respond_sentence

#%%
import numpy as np

char_indices = {}
for i, char in enumerate(chars_list):
    char_indices[char] = i
indices_char = {}
for i, char in enumerate(chars_list):
    indices_char[i] = char
    
n_char = len(chars_list)
max_length_x = 128

def sentence_to_vector(sentence):
    vector = np.zeros((1, max_length_x, n_char), dtype=np.bool)
    for j, char in enumerate(sentence):
        vector[0][j][char_indices[char]] = 1
    return vector

#%%
from keras.models import load_model

encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

def respond(message, beta=5):
    vec = sentence_to_vector(message)  
    state_value = encoder_model.predict(vec)
    y_decoder = np.zeros((1, 1, n_char))  
    y_decoder[0][0][char_indices['\t']] = 1 
    respond_sentence = "" 
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta 
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power)) 
        next_char = indices_char[next_index] 
        
        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break 
            
        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))
        y_decoder[0][0][next_index] = 1

        state_value = h

    return respond_sentence

#%%
bot_name = "賢治bot"
your_name = input("おなまえをおしえてください。:")
print()

print(bot_name + ": " + "こんにちは、" + your_name + "さん。")
message = ""
while message != "さようなら。":
    
    while True:
        message = input(your_name + ": ")
        if not is_invalid(message):
            break
        else:
            print(bot_name + ": ひらがなか、カタカナをつかってください。")
            
    response = respond(message)
    print(bot_name + ": " + response)