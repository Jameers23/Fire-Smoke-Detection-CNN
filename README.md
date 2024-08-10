# Fire and Smoke Detection Using CNN

## Project Overview

This project develops an image classification system to detect fire and smoke using Convolutional Neural Networks (CNN). The system is designed to automatically identify instances of fire and smoke from images, aiding in safety and emergency response.

## Scope

The project includes:
- **Data Collection**: Downloading and labeling images.
- **Data Preprocessing**: Cleaning, augmenting, and preparing the data for model training.
- **Model Development**: Building, training, and evaluating a CNN model.
- **Prediction**: Testing the model with new images.

## Objective

The primary objectives are:
- To classify images as 'fire' or 'non_fire' with high accuracy.
- To minimize false positives and enhance detection speed.
- To develop a robust and adaptable model.

## Technology Requirements

- **Python**: Programming language used.
- **TensorFlow**: Framework for CNN model development.
- **Keras**: High-level API for TensorFlow.
- **Pandas**: Data manipulation library.
- **NumPy**: Numerical operations library.
- **Matplotlib & Seaborn**: Visualization libraries.
- **Plotly**: Interactive graphing library.
- **Kaggle**: Platform for dataset access.
- **Google Colab**: Cloud-based execution environment.

## Code Explanation

### 1. Setup and Data Preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

sns.set_style('darkgrid')

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d phylake1337/fire-dataset
!unzip fire-dataset.zip
```

- **Import Libraries**: Essential libraries for data manipulation, visualization, and model building are imported.
- **Dataset Download**: The dataset is downloaded from Kaggle and unzipped for access.

```python
#create an empty DataFrame
df = pd.DataFrame(columns=['path','label'])

#loop over fire images and label them 1
for dirname, _, filenames in os.walk('/content/fire_dataset/fire_images'):
    for filename in filenames:
        df = df.append(pd.DataFrame([[os.path.join(dirname, filename),'fire']],columns=['path','label']))

#loop over non fire images and label them 0
for dirname, _, filenames in os.walk('/content/fire_dataset/non_fire_images'):
    for filename in filenames:
        df = df.append(pd.DataFrame([[os.path.join(dirname, filename),'non_fire']],columns=['path','label']))

#shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)
df.head(10)
```

- **DataFrame Creation**: An empty DataFrame is initialized to store image paths and labels.
- **Labeling Images**: Images are labeled as 'fire' or 'non_fire' based on their directory.
- **Shuffling**: The dataset is shuffled to randomize the order of images.

### 2. Data Visualization

```python
fig = px.scatter(data_frame=df, x=df.index, y='label', color='label', title='Distribution of fire and non-fire images along the length of the dataframe')
fig.update_traces(marker_size=2)

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "pie"}]])
fig.add_trace(go.Bar(x=df['label'].value_counts().index, y=df['label'].value_counts().to_numpy(), marker_color=['darkorange', 'green'], showlegend=False), row=1, col=1)
fig.add_trace(go.Pie(values=df['label'].value_counts().to_numpy(), labels=df['label'].value_counts().index, marker=dict(colors=['darkorange', 'green'])), row=1, col=2)
```

- **Distribution Visualization**: A scatter plot and bar chart visualize the distribution of 'fire' and 'non_fire' images.

```python
label = 'fire'
data = df[df['label'] == label]
sns.set_style('dark')

pics = 6
fig, ax = plt.subplots(int(pics//2), 2, figsize=(15, 15))
plt.suptitle('Images with Fire')
ax = ax.ravel()
for i in range((pics//2)*2):
    path = data.sample(1).loc[:, 'path'].to_numpy()[0]
    img = image.load_img(path)
    img = image.img_to_array(img)/255
    ax[i].imshow(img)
    ax[i].axes.xaxis.set_visible(False)
    ax[i].axes.yaxis.set_visible(False)

label = 'non_fire'
data = df[df['label'] == label]
sns.set_style('dark')

pics = 6
fig, ax = plt.subplots(int(pics//2), 2, figsize=(15, 15))
plt.suptitle('Images without Fire')
ax = ax.ravel()
for i in range((pics//2)*2):
    path = data.sample(1).loc[:, 'path'].to_numpy()[0]
    img = image.load_img(path)
    img = image.img_to_array(img)/255
    ax[i].imshow(img)
    ax[i].axes.xaxis.set_visible(False)
    ax[i].axes.yaxis.set_visible(False)
```

- **Image Visualization**: Displays sample images from both 'fire' and 'non_fire' categories.

### 3. Data Analysis

```python
def shaper(row):
    shape = image.load_img(row['path']).size
    row['height'] = shape[1]
    row['width'] = shape[0]
    return row
df = df.apply(shaper, axis=1)
df.head(5)

sns.set_style('darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 0.5, 0.5]}, figsize=(15, 10))
sns.kdeplot(data=df.drop(columns=['path', 'label']), ax=ax1, legend=True)
sns.boxplot(data=df, y='height', ax=ax2, color='skyblue')
sns.boxplot(data=df, y='width', ax=ax3, color='orange')
plt.suptitle('Distribution of image shapes')
ax3.set_ylim(0, 7000)
ax2.set_ylim(0, 7000)
plt.tight_layout()
```

- **Image Shape Analysis**: Examines and visualizes the distribution of image dimensions.

### 4. Data Augmentation and Model Preparation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=2,
    zoom_range=0.2,
    rescale=1/255,
    validation_split=0.2,
)

train_gen = generator.flow_from_dataframe(df, x_col='path', y_col='label', target_size=(256, 256), class_mode='binary', subset='training')
val_gen = generator.flow_from_dataframe(df, x_col='path', y_col='label', target_size=(256, 256), class_mode='binary', subset='validation')
```

- **Image Data Augmentation**: Uses `ImageDataGenerator` to apply data augmentation techniques and create training and validation generators.

### 5. Model Building

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

- **CNN Model Architecture**: Defines the CNN model with convolutional layers, max pooling, and dense layers.

### 6. Model Compilation and Training

```python
from tensorflow.keras.metrics import Recall, AUC
from tensorflow.keras.utils import plot_model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), AUC()])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

model.fit(x=train_gen, batch_size=32, epochs=10, validation_data=val_gen, callbacks=[early_stopping, reduce_lr_on_plateau])

history = model.history.history
px.line(history, title="Metrics Plot")
```

- **Model Compilation**: Configures the model with optimizer, loss function, and metrics.
- **Training**: Trains the model with callbacks for early stopping and learning rate reduction.

### 7. Model Evaluation and Prediction

```python
eval_list = model.evaluate(val_gen, return_dict

=True)
for metric in eval_list.keys():
    print(metric + f": {eval_list[metric]:.2f}")

#Downloading and predicting new images
!curl https://static01.nyt.com/images/2021/02/19/world/19storm-briefing-texas-fire/19storm-briefing-texas-fire-articleLarge.jpg --output predict.jpg

img = image.load_img('predict.jpg')
img = image.img_to_array(img) / 255
img = tf.image.resize(img, (256, 256))
img = tf.expand_dims(img, axis=0)

prediction = int(tf.round(model.predict(x=img)).numpy()[0][0])
print("The predicted value is:", prediction, "and the predicted label is:", class_indices[prediction])

img1 = image.load_img('/content/800px-Altja_jõgi_Lahemaal.jpg')
img1 = image.img_to_array(img1) / 255
img1 = tf.image.resize(img1, (256, 256))
img1 = tf.expand_dims(img1, axis=0)

prediction = int(tf.round(model.predict(x=img1)).numpy()[0][0])
print("The predicted value is:", prediction, "and the predicted label is:", class_indices[prediction])
```

- **Evaluation**: Assesses model performance on the validation set.
- **Prediction**: Tests the model with new images to verify its predictions.

## Future Extensions

- **Expand Dataset**: Include more diverse images for improved model robustness.
- **Advanced Architectures**: Experiment with more complex CNN architectures or transfer learning.
- **Real-time Detection**: Implement real-time image processing for live detection.
- **Integration**: Integrate with surveillance systems for automatic alerts.
- **Performance Optimization**: Fine-tune hyperparameters and optimize model performance.

## References

- Kaggle Fire Dataset: [Dataset Link](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- TensorFlow Documentation: [TensorFlow](https://www.tensorflow.org/)
- Keras Documentation: [Keras](https://keras.io/)
- Google Colab: [Colab](https://colab.research.google.com/)

## Contact

For any queries, feel free to reach out to me at [jameers2003@gmail.com](mailto:jameers2003@gmail.com).
