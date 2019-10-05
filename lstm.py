import tensorflow as tf
import numpy as np
import random
import sys

path = tf.keras.utils.get_file(
    "nietzsche.txt",
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)

with open(path, encoding="utf-8") as f:
    text = f.read().lower()

#print(text)
#print(len(text))

char = sorted(list(set(text)))
#print(char)
#print(len(char))

char_indices = dict((c, i) for i, c in enumerate(char))
indices_char = dict((i, c) for i, c in enumerate(char))

#print(char_indices['a'], indices_char[char_indices['a']])

max_len = 40
step = 3
sentences = []
next_chars = []

for i in range(0,len(text)-max_len, step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])

#print(len(sentences))

x = np.zeros((len(sentences), max_len, len(char)), dtype=np.bool)
y = np.zeros((len(sentences), len(char)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for j, c in enumerate(sentence):
        x[i, j, char_indices[c]] = 1
    y[i,char_indices[next_chars[i]]] = 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(max_len, len(char)))) #use many LSTM layer (duplicate layers)
model.add(tf.keras.layers.Dense(len(char), activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp = np.exp(preds)
    preds = exp / np.sum(exp)
    prob = np.random.multinomial(1,preds,1)
    return np.argmax(prob)

def on_epoch_end(epoch, _):
    print("\n========== Generating text after epoch %d\n" % epoch)
    start_index = random.randint(0, len(text) - max_len - 1)
    for temperature in [1.2, 1.5]: # use higher temperature
        print("temperature: %f" % temperature)

        generated = ""
        sentence = text[start_index:start_index + max_len]
        generated += sentence
        print("generating with seed: %s" % sentence)
        print("===================================")
        sys.stdout.write(generated)

        for i in range(500):
            x_pred = np.zeros((1, max_len, len(char)))
            for i, c in enumerate(sentence):
                x_pred[0,i,char_indices[c]] = 1

            preds = model.predict(x_pred)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=50, callbacks=[callback])