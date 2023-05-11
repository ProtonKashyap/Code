import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess images
def load_images(path, info_file):
    images = []
    labels = []
    with open(info_file, 'r') as f:
        for line in f.readlines():
                data = line.strip().split()
                if(data[2]!='NORM'):
                    file_name, label = data[0], data[3]
                    img_path = os.path.join(path, file_name + '.pgm')
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (128, 128))
                    images.append(img_resized)
                    labels.append(1 if label == 'M' else 0)
    return np.array(images), np.array(labels)

# Load dataset
path = '../Dataset/all-mias'
info_file = '../Dataset/all-mias/info2.txt'
images, labels = load_images(path, info_file)

# Flatten images
images = images.reshape(images.shape[0], -1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

