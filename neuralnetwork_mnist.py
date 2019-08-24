import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


index = 0
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, index + 1)
        plt.imshow(x_train[index], cmap='gray')
        index += 1
print(y_train[:16])
#plt.show()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(25,activation="sigmoid"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation="sigmoid")
    ]
)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train, epochs=20)

result = model.evaluate(x_train, y_train)
print(result)
result = model.evaluate(x_test, y_test)
print(result)