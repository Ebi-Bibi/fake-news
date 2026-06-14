import os
import pandas as pd
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_training_pipeline():
    print("🚀 Memulai Automated ML Pipeline...")
    
    # --- PERBAIKAN UTAMA: Mengunci lokasi absolut berbasis letak file train.py ---
    # BASE_DIR akan mengarah ke folder 'src'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Jalur data disusun naik satu tingkat dari 'src' lalu masuk ke 'data' atau 'models'
    processed_data_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed', 'cleaned_news.csv'))
    model_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
    
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"❌ Data bersih tidak ditemukan di:\n👉 {processed_data_path}\nJalankan preprocessing dulu!")
        
    print("📦 Membaca dataset bersih...")
    df = pd.read_csv(processed_data_path)
    
    # Menghindari error jika ada baris kosong pada kolom teks atau label
    df['clean_content'] = df['clean_content'].fillna('')
    df = df.dropna(subset=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_content'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("✨ Mengekstrak fitur teks dengan TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("🧠 Melatih model Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    print(f"🎯 Evaluasi Selesai! Akurasi Pipeline: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    print("💾 Menyimpan model dan vectorizer ke folder 'models/'...")
    joblib.dump(model, os.path.join(model_dir, 'fake_news_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print("✅ Pipeline sukses! Model siap dideploy.")

if __name__ == "__main__":
    run_training_pipeline()