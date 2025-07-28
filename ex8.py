import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_ s = ['hello', 'how are you', 'what is your name', 'bye']
target_ s = ['hi', 'i am fine', 'i am a bot', 'goodbye']
target_ s = ['<start> '+ txt + '<end>' for txt in target_ s]

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_ s(input_ s + target_ s)
input_sequences = tokenizer. s_to_sequences(input_ s)
target_sequences = tokenizer. s_to_sequences(target_ s)
input_sequences = pad_sequences(input_sequences, padding='post')
target_sequences = pad_sequences(target_sequences, padding='post')

vocab_size = len(tokenizer.word_index) + 1
max_input_len = input_sequences.shape[1]
max_target_len = target_sequences.shape[1]

embedding_dim = 64
lstm_units = 128

encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(enc_emb)

decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm_outputs, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])

attention_layer = Attention()
con _vector = attention_layer([decoder_lstm_outputs, encoder_outputs])
combined = Concatenate(axis=-1)([con _vector, decoder_lstm_outputs])
decoder_outputs = Dense(vocab_size, activation='softmax')(combined)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
decoder_target_data = np.expand_dims(target_sequences, -1)
model.fit([input_sequences, target_sequences], decoder_target_data, batch_size=2, epochs=200, verbose=0)

encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

dec_state_input_h = Input(shape=(lstm_units,))
dec_state_input_c = Input(shape=(lstm_units,))
enc_output_input = Input(shape=(max_input_len, lstm_units))
dec_emb2 = Embedding(vocab_size, embedding_dim)(decoder_inputs)
dec_lstm_outputs2, state_h2, state_c2 = LSTM(lstm_units, return_sequences=True, return_state=True)(dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
con _vector2 = Attention()([dec_lstm_outputs2, enc_output_input])
concat2 = Concatenate(axis=-1)([con _vector2, dec_lstm_outputs2])
dec_outputs2 = Dense(vocab_size, activation='softmax')(concat2)
decoder_model = Model([decoder_inputs, enc_output_input, dec_state_input_h, dec_state_input_c], [dec_outputs2, state_h2, state_c2])

def generate_response(input_ ):
    input_seq = tokenizer. s_to_sequences([input_ ])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    enc_out, state_h, state_c = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.array([[tokenizer.word_index['<start>']]])
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, enc_out, state_h, state_c], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == '<end>' or sampled_word == '' or len(decoded_sentence.split()) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '
            target_seq = np.array([[sampled_token_index]])
            state_h, state_c = h, c
    return decoded_sentence.strip()
print("Input: how are you")
print("Bot :", generate_response("how are you"))
