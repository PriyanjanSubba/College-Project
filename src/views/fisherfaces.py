import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    face = detect_face(img)
    if face is not None:
        img_resized = cv2.resize(face, (200, 200)) 
        return img_resized.flatten() 
    else:
        return None

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w] 

def load_images(dataset_path):
    X, y = [], []
    label_map = {}
    current_label = 0

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = person
        for file in os.listdir(person_dir):
            if file.endswith(('jpg', 'jpeg', 'png')):
                img = preprocess_image(os.path.join(person_dir, file))
                if img is not None:
                    X.append(img)
                    y.append(current_label)
        current_label += 1
    return np.array(X), np.array(y), label_map

def pca(X):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    covariance_matrix = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    X_pca = np.dot(X_centered, eigenvectors)
    return X_pca, eigenvectors, mean_face

def lda(X_pca, y):
    mean_overall = np.mean(X_pca, axis=0)
    classes = np.unique(y)
    S_w, S_b = np.zeros((X_pca.shape[1], X_pca.shape[1])), np.zeros((X_pca.shape[1], X_pca.shape[1]))

    for cls in classes:
        X_cls = X_pca[y == cls]
        mean_cls = np.mean(X_cls, axis=0)
        S_w += np.dot((X_cls - mean_cls).T, (X_cls - mean_cls))
        mean_diff = (mean_cls - mean_overall).reshape(-1, 1)
        S_b += len(X_cls) * np.dot(mean_diff, mean_diff.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    X_lda = np.dot(X_pca, eigenvectors[:, :len(classes) - 1])
    return X_lda, eigenvectors

def train_classifier(X_lda, y, method='knn'):
    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif method == 'svm':
        clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_lda, y)
    return clf

def predict_image(img_path, pca_eigenvectors, lda_eigenvectors, mean_face, clf):
    img = preprocess_image(img_path)
    if img is None:
        return "No face detected in the image."
    
    img_centered = img - mean_face
    img_pca = np.dot(img_centered, pca_eigenvectors)
    img_lda = np.dot(img_pca, lda_eigenvectors)
    label = clf.predict([img_lda])
    return label

dataset_path = 'path_to_dataset'
X, y, label_map = load_images(dataset_path)

X_pca, pca_eigenvectors, mean_face = pca(X)

X_lda, lda_eigenvectors = lda(X_pca, y)

clf = train_classifier(X_lda, y, method='knn')  

label = predict_image('path_to_new_image.jpg', pca_eigenvectors, lda_eigenvectors, mean_face, clf)
print(f"Predicted Label: {label_map[label[0]]}")
