from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from summarizer import extract_text, split_into_sentences, textrank_summary

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    try:
        text = extract_text(filepath)
        sentences = split_into_sentences(text)
        if len(sentences) < 8:
            summary = sentences
        else:
            summary = textrank_summary(sentences, num_sentences=8)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)
