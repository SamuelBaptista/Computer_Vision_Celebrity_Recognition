import cv2
import os

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

import skimage
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
import numpy as np



def load_data(data_dir):
    """Carrega um conjunto de dados e retorna duas listas:
    
    images: Uma lista de arrays Numpy, cada uma representando uma imagem.
    labels: Uma lista de nomes que representam as etiquetas das imagens.
    """
    # Obter todos os subdiretórios do data_dir. Cada um representa um rótulo.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Percorre os diretórios de labels e coleta os dados nas duas listas, labels e imagens
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir)]
        
        # Para cada label, carrega as imagens e adiciona-as à lista de imagens.
        # E adiciona o nome do artista (ou seja, o nome do diretório) à lista de labels.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(str(d))
            
    return images, labels


def display_images_and_labels(images, labels, axis='off'):
    """Imprime uma foto de cada artista no dataset:
    
    *Utilizar um batchsize multiplo de 5 para que a plotagem fique simétrica.
    *O argumento axis pode variar entre 'on' e 'off' para mostar o grid das fotos
    """
    # Cria uma lista com os valores únicos dos labels
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    # Loop para criação de cada objeto com a imagem
    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(len(unique_labels)/5, 5, i)  
        plt.axis(axis)
        plt.title(f"{label}")
        i += 1
        _ = plt.imshow(image)
    plt.show()


def display_label_images(images, labels, label, limit=10, axis='off'):
    """Imprime fotos de um único artista:
    
    *Utilizar um batchsize multiplo de 5 para que a plotagem fique simétrica.
    *O argumento axis pode variar entre 'on' e 'off' para mostar o grid das fotos
    """
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(limit/5, 5, i)  
        plt.axis(axis)
        i += 1
        plt.imshow(image, cmap='gray')
    plt.show()
    
    
def transform_images(images, labels=None, size=256, scale=1.1, nn=8):
    """
    * Detecta a face de cada imagem em uma lista 
    * Faz o crop da face para manter apenas essa informação
    * Converte uma lista de imagens para escala de cinza
    * Faz um resize para o tamanho desejado onde a largura e altura são iguais
    * Coleta os labels das imagens que foram convertidas corretamente 
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    indice1 = []
    _ = [indice1.append(i) for i in range(len(images)) if len(images[i].shape) == 3]
    
    images2 = [images[indice] for indice in indice1]
    labels2 = [labels[indice] for indice in indice1]
    
    indice2 = []
    _ = [indice2.append(i) for i in range(len(images2)) if images2[i].shape[2] == 3]

    images3 = [images2[indice] for indice in indice2]
    labels3 = [labels2[indice] for indice in indice2]
    
    images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images3]
    
    faces = [face_cascade.detectMultiScale(image, scale, nn) for image in images_gray]
    
    face_image = []
    face_labels = []

    for i in range(len(faces)):
        if faces[i] == ():
            continue
        x, y, w, h = faces[i][0]
        face_image.append(images_gray[i][y:y+h, x:x+w])
        face_labels.append(labels3[i])
    
    images_transformed = [skimage.transform.resize(image, (size, size)) for image in face_image]
    
    return images_transformed, face_labels



def create_cnn_model():
    '''Cria a rede neural e carrega os pesos do modelo treinado'''

    cnn = Sequential()

    cnn.add(Conv2D(filters = 16, kernel_size = 2, activation = 'relu', input_shape = (256, 256, 1)))
    # Primeira camada de pooling trazendo o valor máximo a cada matriz quadrada de 4 pixels percorrendo toda a imagem.
    cnn.add(MaxPooling2D(pool_size = 2))
    # Segunda camada de convolução com 32 filtros e um kernel 2x2
    cnn.add(Conv2D(filters = 32, kernel_size = 2, activation = 'relu'))
    # Segunda camada de pooling
    cnn.add(MaxPooling2D(pool_size=2))
    # Terceira camada de convolução com 64filtros e um kernel 2x2
    cnn.add(Conv2D(filters = 64, kernel_size = 2, activation = 'relu'))
    # Terceira camada de pooling
    cnn.add(MaxPooling2D(pool_size=2))
    # Transformando a matriz em um vetor para input na primeira camada densa
    cnn.add(Flatten())
    # Primeira camada densa com 512 neurônios
    cnn.add(Dense(512, activation = 'relu'))
    # Segunda camada densa com 256 neurônios
    cnn.add(Dense(256, activation = 'relu'))
    # Camada de Dropout que desativa aleatóriamente 50% dos neurônios a cada passada.
    # Esse processo força a rede à utilizar diferentes atributos para realizar as previsôes e melhora a generalização.
    cnn.add(Dropout(0.5))
    # Camada de saída com o número de posições igual ao número de rótulos para definir uma probabilidade associada a cada um deles.
    cnn.add(Dense(15, activation = 'softmax'))
    
    cnn.load_weights('cnn_model.hdf5')
    
    return cnn



def predict_label(img):
    '''faz o pré processamento e realiza a previsão em uma imagem'''
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = skimage.data.imread(img)

    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            face = face_cascade.detectMultiScale(image2, 1.1, 16)
            
            if face == ():
                face_image = image2
            else:
                x, y, w, h = face[0]
                face_image = image2[y:y+h, x:x+w]
    else:
            face = face_cascade.detectMultiScale(image, 1.1, 16)
            
            if face == ():
                face_image = rgb2gray(image)
            else:
                x, y, w, h = face[0]
                face_image = image[y:y+h, x:x+w]
                face_image = rgb2gray(face_image)
           
    face_resize = skimage.transform.resize(face_image, (256, 256))

    cnn = create_cnn_model()
    
    predicted = cnn.predict(face_resize.reshape(1,256,256,1))
    
    encoder = OneHotEncoder(sparse=False, categories='auto')
    
    labels = np.array(['Aaron Taylor Johnson', 'Adele', 'Aaron Judge', 'Alan Alda', 'Adam Sandler', 'Adriana Lima',
              'Adrianne Palicki', 'Al Pacino', 'Aaron Paul', 'Alan Rickman', 'Adriana Barraza', 'Abigail Breslin',
              'Alan Arkin', 'Adrien Brody', 'Al Roker'])
    
    encoder.fit_transform(labels.reshape(len(labels), 1))
    
    name = encoder.inverse_transform(predicted)
    
    return name[0][0]


def crop_face(img):
    '''faz o pré processamento e retorna uma face'''
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = skimage.data.imread(img)

    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            face = face_cascade.detectMultiScale(image2, 1.1, 16)
            
            if face == ():
                face_image = image2
            else:
                x, y, w, h = face[0]
                face_image = image2[y:y+h, x:x+w]
    else:
            face = face_cascade.detectMultiScale(image, 1.1, 16)
            
            if face == ():
                face_image = rgb2gray(image)
            else:
                x, y, w, h = face[0]
                face_image = image[y:y+h, x:x+w]
                face_image = rgb2gray(face_image)
           
    face_resize = skimage.transform.resize(face_image, (256, 256))
        
    return face_resize


def predict_face(img):
    '''faz a previsão em uma imagem já pre processada'''
      
    cnn = create_cnn_model()
        
    predicted = cnn.predict(img.reshape(1,256,256,1))
    
    encoder = OneHotEncoder(sparse=False, categories='auto')
    
    labels = np.array(['Aaron Taylor Johnson', 'Adele', 'Aaron Judge', 'Alan Alda', 'Adam Sandler', 'Adriana Lima',
              'Adrianne Palicki', 'Al Pacino', 'Aaron Paul', 'Alan Rickman', 'Adriana Barraza', 'Abigail Breslin',
              'Alan Arkin', 'Adrien Brody', 'Al Roker'])
    
    encoder.fit_transform(labels.reshape(len(labels), 1))
    
    name = encoder.inverse_transform(predicted)
    
    return name[0][0]



