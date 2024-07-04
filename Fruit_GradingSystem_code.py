import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
train_dir = "C:/Users/avi11/Downloads/archive (6)/dataset/train"
test_dir = 'C:/Users/avi11/Downloads/archive (6)/dataset/test'
train_datagen =
tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen =
tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(64, 64),
batch_size=32,
class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(64, 64),
batch_size=32,
class_mode='categorical')
Found 10901 images belonging to 2 classes.
Found 2698 images belonging to 2 classes.
model = Sequential([
Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,
64, 3)),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(
train_generator,

steps_per_epoch=train_generator.samples // 32,
epochs=5,
validation_data=test_generator,
validation_steps=test_generator.samples // 32)
Epoch 1/5
340/340 [==============================] - 194s 565ms/step - loss:
0.4015 - accuracy: 0.8014 - val_loss: 0.2300 - val_accuracy: 0.9018
Epoch 2/5
340/340 [==============================] - 67s 197ms/step - loss:
0.2180 - accuracy: 0.9085 - val_loss: 0.1607 - val_accuracy: 0.9390
Epoch 3/5
340/340 [==============================] - 70s 205ms/step - loss:
0.1602 - accuracy: 0.9358 - val_loss: 0.1667 - val_accuracy: 0.9379
Epoch 4/5
340/340 [==============================] - 69s 202ms/step - loss:
0.1126 - accuracy: 0.9544 - val_loss: 0.1672 - val_accuracy: 0.9356
Epoch 5/5
340/340 [==============================] - 68s 199ms/step - loss:
0.0912 - accuracy: 0.9646 - val_loss: 0.1002 - val_accuracy: 0.9576
<keras.callbacks.History at 0x24fb5328f90>
history = model.fit(
train_generator,
steps_per_epoch=train_generator.samples // 32,
epochs=5,
validation_data=test_generator,
validation_steps=test_generator.samples // 32)
Epoch 1/5
340/340 [==============================] - 66s 195ms/step - loss
0.0696 - accuracy: 0.9733 - val_loss: 0.0827 - val_accuracy: 0.9673
Epoch 2/5
340/340 [==============================] - 61s 178ms/step - loss
0.0486 - accuracy: 0.9811 - val_loss: 0.0702 - val_accuracy: 0.9717
Epoch 3/5
340/340 [==============================] - 64s 189ms/step - loss
0.0400 - accuracy: 0.9856 - val_loss: 0.0871 - val_accuracy: 0.9665
Epoch 4/5
340/340 [==============================] - 65s 190ms/step - loss
0.0396 - accuracy: 0.9849 - val_loss: 0.0672 - val_accuracy: 0.9762
Epoch 5/5
340/340 [==============================] - 64s 189ms/step - loss
0.0323 - accuracy: 0.9882 - val_loss: 0.0571 - val_accuracy: 0.9784
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
model.save('model.h5')
import tkinter as tk from
tkinter import filedialog from
PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
image_height, image_width = 64, 64
def preprocess_image(image_path): image =
Image.open(image_path).resize((image_width, image_height))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)
return image_array, image
def predict_fruit_grade(model): file_path =
filedialog.askopenfilename(initialdir="/", title="Select
Image File",
filetypes=(("Image files",
"*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")))
if file_path: image_array, image =
preprocess_image(file_path)
prediction = model.predict(image_array)

class_labels = ['fresh', 'rotten']
if prediction.size > 0 and len(class_labels) ==
prediction.shape[1]: predicted_class =
class_labels[np.argmax(prediction)]
result_label.config(text=f"Fruit Grade: {predicted_class}",
fg="green", font=("Arial", 18, "bold"))
# Display the image on the UI photo =
ImageTk.PhotoImage(image) image_label.config(image=photo)
image_label.image = photo else:
result_label.config(text="Invalid predictions or mismatch in class
labels.", fg="red", font=("Arial", 12))
image_label.config(image="") else:
result_label.config(text="No image selected.", fg="blue",
font=("Arial", 12))
image_label.config(image="")
model = load_model('model.h5')
window = tk.Tk()
window.title("Fruit Grading System")
window.geometry("400x400") window.configure(bg='pink')
browse_button = tk.Button(window, text="Browse Image", command=lambda:
predict_fruit_grade(model),font=("Arial", 14), bg="lightblue",
fg="black")
browse_button.pack(pady=20)
result_label = tk.Label(window, text="", font=("Arial", 16),
bg='white') result_label.pack()
image_label = tk.Label(window, bg='white')
image_label.pack() window.mainloop()
1/1 [==============================] - 0s 171ms/step
1/1 [==============================] - 0s 35ms/step
1/1 [==============================] - 0s 28ms/step
1/1 [==============================] - 0s 23ms/step
1/1 [==============================] - 0s 30ms/step
1/1 [==============================] - 0s 24ms/step
1/1 [==============================] - 0s 22ms/step
