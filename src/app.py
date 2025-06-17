# 1. Kötü Amaçlı Yazılım İkili Dosyasını Gri Tonlamalı Görüntüye Dönüştürme ve CNN Sınıflandırması için Python Kodu
# Bu örnek, kötü amaçlı yazılım ikili dosyalarını gri tonlamalı görüntülere dönüştürme ve ardından bu görüntüleri kullanarak 
# bir Evrişimli Sinir Ağı (CNN) modeli oluşturma ve eğitme sürecini göstermektedir. 
# Bu teknik, kötü amaçlı yazılımın görsel özelliklerini analiz ederek tespiti amaçlamaktadır.

import os
import array
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import LeakyReLU
from keras import backend as K

# Keras backend ayarı (TensorFlow veya Theano için)
K.set_image_data_format('channels_first') # veya 'channels_last'

# --- 1. Kötü Amaçlı Yazılım İkili Dosyasını Gri Tonlamalı Görüntüye Dönüştürme Fonksiyonu ---
def convert_binary_to_grayscale_image(filename, width=256):
    """
    Kötü amaçlı yazılım ikili dosyasını gri tonlamalı görüntüye dönüştürür.
    Her bayt bir piksel değeri olarak yorumlanır (0-255).
    """
    try:
        with open(filename, 'rb') as f:
            ln = os.path.getsize(filename)
            rem = ln % width
            a = array.array("B")
            a.fromfile(f, ln - rem)
        
        # Görüntü boyutunu ayarla
        height = len(a) // width
        if height == 0: # Dosya çok küçükse veya boşsa
            return None

        g = np.reshape(a, (height, width))
        g = np.uint8(g) # 0-255 aralığında unsigned integer olarak ayarla

        # Görüntüyü kaydet (isteğe bağlı)
        # img_output_path = f"{os.path.splitext(filename)}.png"
        # Image.fromarray(g).save(img_output_path)
        # print(f"Görüntü kaydedildi: {img_output_path}")

        return g

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

# --- 2. CNN Modeli Oluşturma ve Eğitme Fonksiyonu ---
def build_and_train_cnn_model(train_images, train_labels, test_images, test_labels, num_classes, img_width, img_height, epochs=20, batch_size=64):
    """
    Kötü amaçlı yazılım görüntüleri için bir CNN modeli oluşturur ve eğitir.
    """
    # Görüntüleri CNN girişi için yeniden şekillendir
    # channels_first için (num_samples, channels, height, width)
    # channels_last için (num_samples, height, width, channels)
    if K.image_data_format() == 'channels_first':
        train_images = train_images.reshape(train_images.shape, 1, img_height, img_width)
        test_images = test_images.reshape(test_images.shape, 1, img_height, img_width)
        input_shape = (1, img_height, img_width)
    else:
        train_images = train_images.reshape(train_images.shape, img_height, img_width, 1)
        test_images = test_images.reshape(test_images.shape, img_height, img_width, 1)
        input_shape = (img_height, img_width, 1)

    # Piksel değerlerini normalize et (0-1 aralığına)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Etiketleri one-hot encode et
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print("CNN modeli eğitiliyor...")
    history = model.fit(train_images, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_images, test_labels))
    
    score = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test doğruluğu: {score[1]*100:.2f}%')
    return model, history

# --- Kullanım Örneği (Örnek Veri ile) ---
if __name__ == "__main__":
    # Gerçek kötü amaçlı yazılım ikili dosyaları yerine örnek veri oluşturma
    # Normalde, kötü amaçlı yazılım örneklerini bir dizinden okursunuz
    # ve her birini convert_binary_to_grayscale_image fonksiyonuyla işlersiniz.
    
    # Örnek: 100x100 boyutunda 100 adet rastgele gri tonlamalı görüntü oluştur
    # ve 2 sınıf (kötü amaçlı/iyi amaçlı) için etiketler ata
    num_samples = 100
    img_height = 100
    img_width = 100
    num_classes = 2 # Örneğin, 0: İyi amaçlı, 1: Kötü amaçlı

    # Rastgele veri oluşturma (gerçek kötü amaçlı yazılım verisi yerine)
    # Normalde, convert_binary_to_grayscale_image fonksiyonunu kullanarak gerçek ikili dosyaları işlersiniz.
    train_images = np.random.randint(0, 256, size=(num_samples, img_height, img_width), dtype=np.uint8)
    train_labels = np.random.randint(0, num_classes, size=(num_samples,))
    test_images = np.random.randint(0, 256, size=(num_samples // 4, img_height, img_width), dtype=np.uint8)
    test_labels = np.random.randint(0, num_classes, size=(num_samples // 4,))

    print(f"Eğitim görüntüleri boyutu: {train_images.shape}")
    print(f"Eğitim etiketleri boyutu: {train_labels.shape}")
    print(f"Test görüntüleri boyutu: {test_images.shape}")
    print(f"Test etiketleri boyutu: {test_labels.shape}")

    # Modeli eğit
    cnn_model, cnn_history = build_and_train_cnn_model(
        train_images, train_labels, test_images, test_labels,
        num_classes, img_width, img_height
    )

    # Model özetini göster
    cnn_model.summary()

    # Gerçek bir ikili dosyayı dönüştürme örneği (örnek dosya oluşturulur)
    # Bu kısmı kendi kötü amaçlı yazılım örneklerinizle değiştirebilirsiniz.
    dummy_malware_file = "dummy_malware.bin"
    with open(dummy_malware_file, "wb") as f:
        f.write(os.urandom(1024 * 50)) # 50 KB'lık rastgele veri

    print(f"\n'{dummy_malware_file}' ikili dosyasını gri tonlamalı görüntüye dönüştürme:")
    malware_image = convert_binary_to_grayscale_image(dummy_malware_file, width=256)
    if malware_image is not None:
        print(f"Dönüştürülen görüntü boyutu: {malware_image.shape}")
        # Görüntüyü göstermek için matplotlib kullanabilirsiniz
        # import matplotlib.pyplot as plt
        # plt.imshow(malware_image, cmap='gray')
        # plt.title(f"'{dummy_malware_file}' Görüntüsü")
        # plt.show()
    
    # Oluşturulan örnek dosyayı temizle
    os.remove(dummy_malware_file)