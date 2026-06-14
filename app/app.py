import os
import sys
from flask import Flask, request, jsonify
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_text

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

try:
    raw_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    
    model.classes_ = raw_model.classes_
    model.coef_ = raw_model.coef_
    model.intercept_ = raw_model.intercept_
    if hasattr(raw_model, 'n_features_in_'):
        model.n_features_in_ = raw_model.n_features_in_
        
    print("🎯 Model and Vectorizer loaded and FIXED successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, vectorizer = None, None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk memastikan API berjalan dengan baik"""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint utama untuk memprediksi berita hoaks atau bukan"""
    if not request.is_json:
        return jsonify({"error": "Invalid input format. Request body must be JSON."}), 400
    
    data = request.get_json()
    news_text = data.get('text', '')
    
    if not news_text or not news_text.strip():
        return jsonify({"error": "Missing 'text' field or the text is empty."}), 400
    
    if model is None or vectorizer is None:
        return jsonify({"error": "Model machine learning tidak tersedia di server."}), 500

    try:
        cleaned_text = clean_text(news_text)
        
        transformed_text = vectorizer.transform([cleaned_text])
        
        prediction = int(model.predict(transformed_text)[0])
        probability = model.predict_proba(transformed_text)[0][prediction]
        
        result = "Fake News" if prediction == 1 else "Real News"
        return jsonify({
            "status": "success",
            "prediction": result,
            "label": prediction,
            "confidence": round(float(probability), 4),
            "text_preview": news_text[:100] + "..."
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat pemrosesan: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)