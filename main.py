import os
import pandas as pd
from google.oauth2 import service_account
import gspread
from google.cloud import aiplatform
from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

app = Flask(__name__)

# Set up credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly', 'https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'path/to/your/service_account_key.json'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Connect to Google Sheets and Drive
sheets_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)

# Set up Vertex AI
PROJECT_ID = 'your-project-id'
LOCATION = 'us-central1'
MODEL_ID = 'text-bison@001'

aiplatform.init(project=PROJECT_ID, location=LOCATION)

def get_sheet_data(sheet_id):
    sheet = sheets_client.open_by_key(sheet_id).sheet1
    data = sheet.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0])

def get_folder_data(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and (mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/vnd.google-apps.document')",
        fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])
    
    all_data = []
    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            df = get_sheet_data(item['id'])
            all_data.append(f"Sheet: {item['name']}\n{df.to_string()}")
        elif item['mimeType'] == 'application/vnd.google-apps.document':
            doc = drive_service.files().export(fileId=item['id'], mimeType='text/plain').execute()
            all_data.append(f"Document: {item['name']}\n{doc.decode('utf-8')}")
    
    return "\n\n".join(all_data)

def predict_large_language_model_sample(
    project_id: str,
    model_id: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    model = aiplatform.TextGenerationModel.from_pretrained(model_id)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response.text

def generate_prompt(question, data):
    context = f"Here's the data from the Google Drive:\n{data}\n\n"
    prompt = f"{context}Question: {question}\nAnswer:"
    return prompt

def chatbot(question, data):
    prompt = generate_prompt(question, data)
    response = predict_large_language_model_sample(
        project_id=PROJECT_ID,
        model_id=MODEL_ID,
        temperature=0.2,
        max_decode_steps=256,
        top_p=0.8,
        top_k=40,
        content=prompt,
    )
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    drive_link = request.json['drive_link']
    if 'folders' in drive_link:
        folder_id = drive_link.split('/')[-1]
        data = get_folder_data(folder_id)
    elif 'spreadsheets' in drive_link:
        sheet_id = drive_link.split('/')[-2]
        data = get_sheet_data(sheet_id).to_string()
    else:
        return jsonify({'error': 'Invalid link. Please provide a Google Drive folder or Google Sheet link.'})
    
    return jsonify({'message': 'Data loaded successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    drive_link = request.json['drive_link']
    
    if 'folders' in drive_link:
        folder_id = drive_link.split('/')[-1]
        data = get_folder_data(folder_id)
    elif 'spreadsheets' in drive_link:
        sheet_id = drive_link.split('/')[-2]
        data = get_sheet_data(sheet_id).to_string()
    else:
        return jsonify({'error': 'Invalid link. Please provide a Google Drive folder or Google Sheet link.'})
    
    answer = chatbot(question, data)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
