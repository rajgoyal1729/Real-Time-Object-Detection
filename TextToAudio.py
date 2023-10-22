import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pyttsx3

# Define the parameters
NUM_CLASSES = 4  # Number of object classes in COCO dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Create a new model by adding custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess COCO dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'datasets/coco',  # Replace with the path to your COCO dataset
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'datasets/coco',  # Replace with the path to your COCO dataset
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=1)

# Load and preprocess an image for object detection
sample_image_path = 'datasets/coco/images/val2017/000000183648.jpg'  # Replace with your image
sample_image = load_img(sample_image_path, target_size=IMG_SIZE)
sample_image_array = img_to_array(sample_image)
sample_image_array = sample_image_array / 255.0  # Normalize pixel values

# Make predictions on the sample image
predictions = model.predict(tf.convert_to_tensor([sample_image_array]))

# Map class indices to class labels (e.g., 'cat', 'dog')
class_labels = ['class1', 'class2', ...]  # Define your class labels

# Convert class indices to labels
predicted_labels = [class_labels[predictions.argmax()]]

# Convert the detected objects to text
detected_objects = ', '.join(predicted_labels)
# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Use the 'say' function to pronounce the detected objects
engine.say(f'Detected objects: {detected_objects}')
engine.runAndWait()

