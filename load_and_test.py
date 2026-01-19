from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model
model = load_model("pneumonia_cnn_model.h5")

# Setup test data generator (same as before)
test_dir = 'chest_xray/test'
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Predict using the model
y_pred = (model.predict(test_data) > 0.5).astype("int32")
y_true = test_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

