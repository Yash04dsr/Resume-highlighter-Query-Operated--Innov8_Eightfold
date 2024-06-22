import os
import tempfile
import fitz
from flask import Flask, render_template, request, make_response, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Initialize models
oracle = pipeline("question-answering", model="deepset/roberta-base-squad2")
theta = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Recurring global valueholders
file_name_counter = 1
temp_dir = None
temp_file_path = None
text = None
printable = set(list(' ,`0123456789-=~!@#$%^&*()_+abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,:;"\'/\n[]\\{}|?><.'))
bloat = set(('per', 'and', 'but', 'the', 'for', 'are', 'was', 'were', 'be', 'been', 'with', 'you', 'this', 'but', 'his', 'from', 'they', 'say', 'her', 'she', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'out', 'about', 'who', 'get', 'which', 'when', 'make', 'can', 'like', 'time', 'just', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most'))

# Helper Functions
def phrases_by_relevance(text, prompt):
    qembed = theta.encode([prompt])
    embeddings = theta.encode(text)
    result = [tensor.item() for tensor in list(cos_sim(qembed, embeddings)[0])]
    phraset = [(text[i], result[i]) for i in range(len(text))]
    phraset.sort(key=lambda x: x[1], reverse=True)
    return phraset

def cleaned(t):
    return "".join([i for i in t if i in printable])

def pdf_to_text(pdf):
    global text
    with fitz.open(pdf) as doc:
        ina = [cleaned(page.get_text()).split("\n") for page in doc]
        texta = []
        for a in ina:
            texta += a
        text = "This is the resume of a person." + " ".join(texta)
        print("found text:", texta)
    return [k for k in texta if len(k) > 2 and not k.isspace() and k not in bloat]

def delimiter(phraset):
    l0 = 1
    for i in range(3):
        if phraset[i][1] > 0.62:
            l0 += 1
    l1 = l0 + 1
    for i in range(10):
        if phraset[l0 + i][1] > 0.60:
            l1 += 1
    l2 = l1 + 1
    for i in range(20):
        if phraset[l1 + i][1] >= 0.52:
            l2 += 1

    i1 = [x[0] for x in phraset[:l0]]
    i2 = [x[0] for x in phraset[l0:l1]]
    i3 = [x[0] for x in phraset[l1:l2]]
    return [i1, i2, i3]

def highlight_pdf(phraset, og, trans):
    colors = [(1.0, 0.8, 0.5), (1.0, 0.9, 0.5), (1.0, 1.0, 0.5)]
    doc = fitz.open(og)
    for page_num in range(len(doc)):
        page = doc[page_num]
        page.clean_contents()

        print(phraset, og, trans)

        for i in range(3):
            for text in phraset[i]:
                print(text)
                text_instances = page.search_for(text)
                for inst in text_instances:
                    try:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=colors[i])
                        highlight.update()
                    except ValueError:
                        print(f"Error occurred while highlighting: for {text} {ValueError}")
                        continue
                kk = text.split()
                for part in kk:
                    if part not in bloat and len(part) > 3:
                        text_instances = page.search_for(part)
                        for inst in text_instances:
                            try:
                                highlight = page.add_highlight_annot(inst)
                                highlight.set_colors(stroke=colors[i])
                                highlight.update()
                            except ValueError:
                                print(f"Error occurred while highlighting: for {part} {ValueError}")
                                continue
    doc.save(trans)
    doc.close()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/get_pdf_data/<path:pdf_file>', methods=['GET'])
def get_pdf_data(pdf_file):
    try:
        with open(pdf_file, 'rb') as f:
            pdf_data = f.read()
        response = make_response(pdf_data)
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'inline; filename="%s"' % os.path.basename(pdf_file))
        return response
    except Exception:
        return jsonify({'error': str(Exception)}), 500
"""
@app.route('/process_question', methods=['POST'])
def process_question():
    global text
    question = request.get_json().get('question')
    if question and text:
        print("Question and text received")
        answer = oracle(question=question, context=text, top_k=1)['answer']
        return jsonify({'answer': answer})
    return jsonify({'error': 'Invalid request'}), 400"""
@app.route('/process_question', methods=['POST'])
def process_question():
    global text
    question = request.get_json().get('question')
    if question and text:
        print("Question and text received")
        answers = oracle(question=question, context=text, top_k=3)
        combined_answer = ""
        for answer in answers:
            combined_answer += answer['answer'] + ". "  # Concatenate answers with a period and space
        return jsonify({'answer': combined_answer})
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global file_name_counter, temp_dir, temp_file_path, text
    pdf_file = request.files['pdf_file']
    prompt = request.form['prompt']
    if pdf_file and prompt:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, pdf_file.filename)
        pdf_file.save(temp_file_path)

        # Convert PDF to text
        context = pdf_to_text(temp_file_path)
        modified_pdf_files = []

        # Finding key phrases
        phraset = phrases_by_relevance(context, prompt)
        print('Passing these relevant texts to delimiter:', phraset)
        classified = delimiter(phraset)
        print('importance order:', classified)

        modified_pdf_file = os.path.join(temp_dir, f'Highlighted_{file_name_counter}.pdf')
        highlight_pdf(classified, temp_file_path, modified_pdf_file)
        modified_pdf_files.append(modified_pdf_file)
        file_name_counter += 1

        return jsonify({'modified_pdf_files': modified_pdf_files})
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
