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


#2. API Çağrı Dizileriyle Kötü Amaçlı Yazılım Tespiti için Çizge Evrişimli Ağ (GCN) Python Kodu

# Bu örnek, API çağrı dizilerini kullanarak kötü amaçlı yazılım tespiti için basit bir Çizge Evrişimli Ağ (GCN) uygulamasını
# göstermektedir. API çağrı dizileri, bir çizge yapısına dönüştürülerek GCN tarafından işlenir.

import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from math import ceil
import sys
import warnings

warnings.filterwarnings('ignore')

# Sabitler
SEED = 42
NB_SAMPLES = 43876 # Örnek veri kümesi boyutu [4]
VOCAB_SIZE = 307 # API çağrılarının benzersiz sayısı (0-306) [4]
MAX_LENGTH = 100 # Her API çağrı dizisindeki maksimum tekrar etmeyen çağrı sayısı [4]
NB_CLASSES = 2 # 0: İyi amaçlı, 1: Kötü amaçlı [4]
WORD_EMBED_DIM = 5 # Kelime gömme boyutu (API çağrıları için)
GRAPH_DIM = 10 # Çizge gömme boyutu

# Tek sıcak kodlama yardımcı fonksiyonu
def one_hot_encode(y):
    mods = len(np.unique(y))
    y_enc = np.zeros((y.shape, mods))
    for i in range(y.shape):
        y_enc[i, y[i]] = 1
    return y_enc

# Çizge özelliklerini alma yardımcı fonksiyonu
def get_graph_features(G, all_embds, nmax=MAX_LENGTH):
    n = len(G.nodes())
    adj = np.zeros((nmax, nmax))
    embds = np.zeros((nmax, all_embds.shape[1]))
    node2id = {node: i for i, node in enumerate(G.nodes())}

    for i in G.nodes():
        if node2id[i] < nmax:
            embds[node2id[i]] = all_embds[i] # API çağrısı ID'sini gömme olarak kullan
        for j in G.neighbors(i):
            if node2id[i] < nmax and node2id[j] < nmax:
                adj[node2id[j], node2id[i]] = 1
    return adj, embds

# Örnek çizge simülasyonu (gerçek API çağrı verisi yerine)
def simulate_graph(nb_samples=NB_SAMPLES, vocab_size=VOCAB_SIZE, graph_length=MAX_LENGTH, word_embed_dim=WORD_EMBED_DIM, nb_classes=NB_CLASSES, random_state=None):
    np.random.seed(random_state)
    adjs =
    embds =
    y = np.zeros(nb_samples)

    # API çağrısı ID'leri için basit gömmeler oluştur
    all_embds = np.random.normal(size=(vocab_size, word_embed_dim))

    for i in range(nb_samples):
        # Rastgele sınıf ata
        cat = np.random.randint(0, nb_classes)
        y[i] = cat

        # Rastgele bir API çağrı dizisi oluştur
        api_sequence = np.random.randint(0, vocab_size, size=(graph_length,)).tolist()
        
        # Basit bir yönlendirilmiş çizge oluştur (API çağrı dizisinden)
        G = nx.DiGraph()
        for k in range(len(api_sequence) - 1):
            G.add_edge(api_sequence[k], api_sequence[k+1])
        
        # Çizge özelliklerini al
        adj_matrix, node_embeddings = get_graph_features(G, all_embds, nmax=graph_length)
        adjs.append(adj_matrix)
        embds.append(node_embeddings)
    
    return np.array(adjs), np.array(embds), y

# --- GCN Modeli Tanımı ---
class GCN():
    def __init__(self, node_dim=WORD_EMBED_DIM, graph_dim=GRAPH_DIM, nb_classes=NB_CLASSES, nmax=MAX_LENGTH, alpha=0.025):
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.nb_classes = nb_classes
        self.nmax = nmax
        self.alpha = alpha
        self.build_model()

    def build_model(self):
        self.adjs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])
        self.embeddings = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nmax, self.node_dim])
        self.targets = tf.compat.v1.placeholder(tf.float32, shape=[None, self.nb_classes])

        # GCN katmanı
        # A_hat = D_hat^-1/2 * A_hat * D_hat^-1/2
        # Burada A_hat, öz-döngülerle birlikte komşuluk matrisidir (I + A)
        # D_hat, A_hat'ın derece matrisidir
        # Basitlik için, doğrudan bir ağırlık matrisi ile çarpma yapıyoruz
        # Daha karmaşık GCN uygulamaları normalleştirme adımlarını içerir
        
        # İlk katman
        W1 = tf.Variable(tf.random.normal([self.node_dim, self.graph_dim]))
        H1 = tf.matmul(self.embeddings, W1) # XW
        H1_agg = tf.matmul(self.adjs, H1) # AHXW
        H1_activated = tf.nn.relu(H1_agg) # ReLU aktivasyonu

        # İkinci katman (çıkış katmanı)
        W2 = tf.Variable(tf.random.normal([self.graph_dim, self.nb_classes]))
        # Çizge seviyesi sınıflandırma için düğüm gömmelerini topla (ortalama havuzlama)
        G_pooled = tf.reduce_mean(H1_activated, axis=1) # Düğüm gömmelerinin ortalaması
        logits = tf.matmul(G_pooled, W2)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.targets))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.cost)
        self.prediction = tf.nn.softmax(logits)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def fit(self, adjs, embds, y, epochs=20, batch_size=10, shuffle=True):
        self.scores =
        y_enc = one_hot_encode(y)
        minibatches = ceil(len(adjs) / batch_size)
        
        for i in range(epochs):
            if shuffle:
                idx = np.random.permutation(len(adjs))
                adjs_shuffled = adjs[idx]
                embds_shuffled = embds[idx]
                y_enc_shuffled = y_enc[idx]
            else:
                adjs_shuffled = adjs
                embds_shuffled = embds
                y_enc_shuffled = y_enc

            mini = np.array_split(np.arange(len(adjs)), minibatches)
            
            for inds in mini:
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={
                    self.adjs: adjs_shuffled[inds],
                    self.embeddings: embds_shuffled[inds],
                    self.targets: y_enc_shuffled[inds]
                })
            
            # Her epoch sonunda doğruluk hesapla
            train_score = self.score(adjs, embds, y)
            self.scores.append(train_score)
            sys.stderr.write(f'\rEpoch: {i+1}/{epochs}, Eğitim Doğruluğu: {train_score:.2f}%')
            sys.stderr.flush()
        print() # Yeni satıra geç

    def score(self, adjs, embds, y):
        y_enc = one_hot_encode(y)
        preds = self.sess.run(self.prediction, feed_dict={
            self.adjs: adjs,
            self.embeddings: embds,
            self.targets: y_enc
        })
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y_enc, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.sess.run(accuracy) * 100

# --- Kullanım Örneği ---
if __name__ == "__main__":
    # TensorFlow 2.x'te uyumluluk için v1 davranışını etkinleştir
    tf.compat.v1.disable_eager_execution()

    # Örnek veri kümesi oluştur (gerçek API çağrı dizileri yerine)
    # Normalde, API çağrı dizilerini bir sandbox ortamından toplarsınız
    # ve bunları çizge temsillerine dönüştürürsünüz.
    # Örneğin, [4]'teki gibi bir CSV dosyasından okuyabilirsiniz:
    # df = pd.read_csv('dynamic_api_call_sequence_per_malware_100_0_306.csv')
    # df['malware'] sütunu etiketleriniz olurdu.
    # API çağrıları (t_0... t_99) düğümler ve aralarındaki ilişkiler kenarlar olurdu.

    print("Örnek veri kümesi oluşturuluyor...")
    ADJ, EMBDS, Y = simulate_graph(NB_SAMPLES, VOCAB_SIZE, MAX_LENGTH, WORD_EMBED_DIM, NB_CLASSES, random_state=SEED)
    
    # Veri dengesizliğini giderme (isteğe bağlı, ancak kötü amaçlı yazılım verilerinde yaygın)
    # [4]'te bahsedilen RandomUnderSampler gibi
    # Bu örnekte, simüle edilmiş veriler zaten dengeli olabilir.
    # random_undersampler = RandomUnderSampler(random_state=SEED)
    # ADJ_resampled, Y_resampled = random_undersampler.fit_resample(ADJ.reshape(ADJ.shape, -1), Y)
    # EMBDS_resampled = EMBDS[random_undersampler.sample_indices_]
    # ADJ_resampled = ADJ_resampled.reshape(ADJ_resampled.shape, MAX_LENGTH, MAX_LENGTH)
    # print(f"Yeniden örneklenen etiket dağılımı: {Counter(Y_resampled)}")
    # ADJ, EMBDS, Y = ADJ_resampled, EMBDS_resampled, Y_resampled

    # Eğitim ve test setlerine ayırma
    SHARE = 0.75
    CUT = int(NB_SAMPLES * SHARE)
    ADJ_train, Y_train, ADJ_test, Y_test = ADJ, Y, ADJ, Y
    EMBDS_train, EMBDS_test = EMBDS, EMBDS

    print(f"Eğitim verisi boyutu: {ADJ_train.shape}, {EMBDS_train.shape}, {Y_train.shape}")
    print(f"Test verisi boyutu: {ADJ_test.shape}, {EMBDS_test.shape}, {Y_test.shape}")

    # GCN modelini başlat ve eğit
    gcn_model = GCN(node_dim=WORD_EMBED_DIM, graph_dim=GRAPH_DIM, nb_classes=NB_CLASSES, nmax=MAX_LENGTH, alpha=0.001)
    gcn_model.fit(ADJ_train, EMBDS_train, Y_train, epochs=15, batch_size=32)

    # Model performansını değerlendir
    test_accuracy = gcn_model.score(ADJ_test, EMBDS_test, Y_test)
    print(f'Test Doğruluğu: {test_accuracy:.2f}%')


#3. LLM Tabanlı Kötü Amaçlı Yazılım Sandbox Raporu Özetleme için Python Kodu (LangChain ile)

#Bu örnek, bir kötü amaçlı yazılım sandbox raporunu özetlemek için Büyük Dil Modellerini (LLM'ler) 
#LangChain çerçevesiyle nasıl kullanabileceğinizi göstermektedir. Bu, analistlerin 
#bilişsel yükünü azaltmaya ve olay müdahalesini hızlandırmaya yardımcı olur. 


import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI # Gemini için
# from langchain_openai import ChatOpenAI # OpenAI için

# API anahtarınızı ayarlayın (örneğin, ortam değişkeni olarak)
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
# veya os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Örnek Kötü Amaçlı Yazılım Sandbox Raporu Oluşturma ---
# Gerçek bir sandbox raporu genellikle JSON veya metin formatında olur.
# Bu örnek için basit bir metin raporu oluşturuyoruz.
sample_report_content = """
Kötü Amaçlı Yazılım Analiz Raporu - Örnek Dosya: malware_sample.exe

MD5: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
SHA256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

Davranış Özeti:
Dosya, C:\\Windows\\System32\\svchost.exe sürecine enjekte edildi.
Kayıt defteri anahtarı HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run altında kalıcılık sağlandı.
Harici IP adresi 192.168.1.100'e (Komuta ve Kontrol sunucusu olduğu düşünülen) HTTP POST isteği gönderildi.
Sistemdeki tüm.doc ve.docx uzantılı dosyalar şifrelendi.
Şifrelenmiş dosyalar için fidye notu (README.txt) bırakıldı.
Dosya sistemi değişiklikleri:
- C:\\Users\\Public\\Documents\\ransom.log oluşturuldu.
- C:\\Users\\Public\\Documents\\README.txt oluşturuldu.
- C:\\Users\\User\\Desktop\\important.doc şifrelendi.
- C:\\Users\\User\\Documents\\report.docx şifrelendi.

Ağ Etkinliği:
- DNS sorgusu: malicious-c2.com
- HTTP isteği: POST http://192.168.1.100/upload_data
- Hedeflenen portlar: 80, 443

API Çağrıları:
- CreateRemoteThread
- WriteProcessMemory
- RegSetValueExA
- HttpSendRequestA
- CryptEncrypt
- CreateFileW

Tespit Edilen Taktikler ve Teknikler (MITRE ATT&CK):
- T1055: Process Injection
- T1547.001: Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder
- T1071.001: Application Layer Protocol: Web Protocols
- T1486: Data Encrypted for Impact
- T1041: Exfiltration Over C2 Channel
"""

# Raporu bir dosyaya yaz
report_file_path = "malware_sandbox_report.txt"
with open(report_file_path, "w", encoding="utf-8") as f:
    f.write(sample_report_content)

# --- 2. LangChain ile Raporu Yükleme ve Bölme ---
loader = TextLoader(report_file_path, encoding="utf-8")
docs = loader.load()

# Metni LLM'in bağlam penceresine sığacak şekilde böl
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

print(f"Oluşturulan belge parçaları sayısı: {len(split_docs)}")

# --- 3. LLM Modelini Başlatma ---
# Google Gemini Pro modelini kullanma
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# OpenAI GPT modelini kullanmak isterseniz:
# llm = ChatOpenAI(model="gpt-4o-mini")

# --- 4. Özetleme Zinciri Oluşturma ---
# Basit "stuff" yaklaşımı: tüm belgeleri tek bir isteme doldurur
# Daha uzun metinler için "map-reduce" veya "refine" stratejileri kullanılabilir [9, 10]
prompt = ChatPromptTemplate.from_messages(
    [("system", "Aşağıdaki kötü amaçlı yazılım analiz raporunun kısa ve öz bir özetini yazın. Ana davranışları, hedefleri ve tespit edilen taktikleri vurgulayın:\n\n{context}")]
)

# Zinciri oluştur
summarization_chain = (
    {"context": lambda docs: "\n\n".join(doc.page_content for doc in docs)}
| prompt
| llm
| StrOutputParser()
)

# --- 5. Özetleme Zincirini Çalıştırma ---
print("\nKötü amaçlı yazılım raporu özetleniyor...")
try:
    summary = summarization_chain.invoke(split_docs)
    print("\n--- Özetlenmiş Kötü Amaçlı Yazılım Raporu ---")
    print(summary)
    print("------------------------------------------")
except Exception as e:
    print(f"Özetleme sırasında bir hata oluştu: {e}")
    print("API anahtarınızın doğru yapılandırıldığından ve modelin erişilebilir olduğundan emin olun.")

# Oluşturulan örnek dosyayı temizle
os.remove(report_file_path)  

