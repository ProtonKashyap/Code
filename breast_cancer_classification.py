# Importing the required libraries
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Define the path to the directory containing the MIAS mammography images
data_dir = '../Dataset/all-mias/'

# Load the image filenames and corresponding labels
with open(os.path.join(data_dir, 'info2.txt')) as f:
    lines = f.readlines()
    filenames = []
    labels = []
    for line in lines:
        data = line.strip().split()
        if len(data) <= 3:
            continue
        else:
            if(data[3]=='M'):
                labels.append(1)
            else:
                labels.append(0)
        filenames.append(data[0] + '.pgm')

# Printing filenames and labels
# for filename in filenames:
#     print(format(filename))
# for label in labels:
#     print(format(label))

#Printing Labels

# Load the images and preprocess them
images = []
for filename in filenames:
    img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = img.flatten()
    images.append(img)
X = np.array(images)
print(len(images))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the KNN classifier on the training data
y = np.array(labels)
knn.fit(X_scaled, y)

# Load a test image and preprocess it
test_img = cv2.imread(os.path.join(data_dir, 'mdb001.pgm'), cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (50, 50))
test_img = test_img.flatten()
test_img_scaled = scaler.transform(np.array([test_img]).reshape(1, -1))

# Predict the class label for the test image
class_label = knn.predict(test_img_scaled)

print(class_label)

# Print the predicted class label
if class_label[0] == 0:
    print('The mammography image is benign.')
else:
    print('The mammography image is malignant.')


from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

k = 5
clf = KNeighborsClassifier(n_neighbors=k)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion matrix:')
print(conf_mat)


#Code 2
# import os
# import numpy as np
# from PIL import Image

# data_dir = '../Dataset/all-mias/'
# img_size = 128

# def load_data():
#     X = []
#     y = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.pgm'):
#             img = Image.open(os.path.join(data_dir, filename))
#             img = img.resize((img_size, img_size), resample=Image.BILINEAR)
#             img = np.asarray(img)
#             X.append(img.flatten())
#             if filename.startswith('M'):
#                 y.append(1)  # malignant
#             else:
#                 y.append(0)  # benign
#     return np.array(X), np.array(y)

# X, y = load_data()
# from sklearn.model_selection import train_test_split

# from sklearn.neighbors import KNeighborsClassifier

# k = 5
# clf = KNeighborsClassifier(n_neighbors=k)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf.fit(X_train, y_train)
# from sklearn.metrics import accuracy_score, confusion_matrix

# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# conf_mat = confusion_matrix(y_test, y_pred)

# print('Accuracy:', accuracy)
# print('Confusion matrix:')
# print(conf_mat)
