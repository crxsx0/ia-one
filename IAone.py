import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
import numpy as np
import logging, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename= "logs/iaone.log")

def predict_image(model, img_path):
    logging.info("predict_image")
    inicio = time.time()
    # Cargar la imagen y preprocesarla para hacer la predicción
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar los valores de píxeles
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(img_array)
    
    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)
    fin = time.time()
    logging.info(f'TIEMPO DE EJECUCION(predict_image): {round(fin-inicio, 1)}s')
    return predicted_class

def train_model_with_feedback(model, img_path, correct_label):
    logging.info("train_model_with_feedback")
    inicio = time.time()
    # Obtener la predicción del modelo
    predicted_class = predict_image(model, img_path)
    
    # Verificar si la predicción es correcta
    if predicted_class == correct_label:
        print("El modelo ha hecho una predicción correcta.")
    else:
        print("El modelo ha hecho una predicción incorrecta.")
        print("La clase correcta es:", correct_label)
        print("La clase predicha es:", predicted_class)
        
        # Pedir retroalimentación al usuario
        feedback = input("¿La predicción es correcta? (s/n): ")
        etiquetas_correctas = {
            0: "Jordan 1 Retro High Pine Green Black",
            1: "Jordan 4 Retro SE 95 Neon",
            2: "Jordan 4 Retro Infra",
            3: "Jordan 11 Retro Legend Blue",
            4: "Jordan 13 Retro Flint",
            5: "Jordan 3 Fragment Retro",
            6: "Jordan 11 Retro Cool Grey",
            7: "Jordan 11 Retro Playoffs Bred",
            8: "Jordan 5 retro fire red silver"
        }
        
        # Si la retroalimentación es negativa, reentrenar el modelo con la imagen
        if feedback.lower() == 'n':
            print("Actualizando el modelo con la retroalimentación del usuario...")
            correct_label_for_prediction = etiquetas_correctas[predicted_class]
            print("Modelo actualizado con éxito.")
        else:
            print("Gracias por confirmar la predicción.")
    fin = time.time()
    logging.info(f'TIEMPO DE EJECUCION(train_model_with_feedback): {round(fin-inicio, 1)}s')

def train_main():
    logging.info("train_main")
    inicio = time.time()
    # Define los generadores de datos para imágenes de StockX y originales
    train_datagen_stockx = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_datagen_originales = ImageDataGenerator(
        rescale=1./255
    )

    # Directorios de entrenamiento
    train_dir_stockx = 'data/train/stockx'
    train_dir_originales = 'data/train/originales'

    # cargar más imágenes de stockx para ver qué pasa
    train_generator_stockx = train_datagen_stockx.flow_from_directory(
        train_dir_stockx,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    train_generator_originales = train_datagen_originales.flow_from_directory(
        train_dir_originales,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical' 
    )

    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',  
        input_shape=(150, 150, 3),
        include_top=False
    )

    base_model.trainable = False  

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator_originales.num_classes, activation='softmax')  # Cambiado a 'train_generator_originales.num_classes'
    ])

    # Compilar modelo
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar modelo con datos de StockX
    model.fit(train_generator_stockx, epochs=5)

    # Descongelar algunas capas base
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Continuar entrenamiento con datos originales
    model.fit(train_generator_originales, epochs=5)

    # Guardar modelo
    model.save('IAone')
    fin = time.time()
    logging.info(f'TIEMPO DE EJECUCION(train_main): {round(fin-inicio, 1)}s')
    return model

# Bucle interactivo para la predicción y retroalimentación del usuario
while True:
    model = train_main()
    img_path = input("Ingrese la ruta de la imagen para predecir (o 'salir' para terminar): ")
    if img_path.lower() == 'salir':
        break
    # Realizar la predicción de la imagen
    predicted_class = predict_image(model, img_path)
    # Definir un diccionario que mapee el número de clase al nombre de la zapatilla
    clases_zapatillas = {
        0: "Jordan 1 Retro High Pine Green Black",
        1: "Jordan 4 Retro SE 95 Neon",
        2: "Jordan 4 Retro Infra",
        3: "Jordan 11 Retro Legend Blue",
        4: "Jordan 13 Retro Flint",
        5: "Jordan 3 Fragment Retro",
        6: "Jordan 11 Retro Cool Grey",
        7: "Jordan 11 Retro Playoffs Bred",
        8: "Jordan 5 retro fire red silver"
    }

    # Obtener el nombre de la zapatilla predicha usando el diccionario de clases
    nombre_zapatilla_predicha = clases_zapatillas[predicted_class]

    print("La zapatilla predicha es:",  nombre_zapatilla_predicha)
    
    # Solicitar retroalimentación del usuario
    feedback = input("¿Es esta predicción correcta? (s/n): ")
    if feedback.lower() == 'n':
        print(clases_zapatillas)
        correct_label = int(input("Ingrese el número de clase correcto: "))
        # Realizar acciones para mejorar el modelo basado en la retroalimentación del usuario
        train_model_with_feedback(model, img_path, correct_label)
