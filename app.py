import os
import io
import time
import openai
import sys
from quart import Quart, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")

# Azure Form Recognizer setup
endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

app = Quart(__name__)

# Function to capture the output from Azure Form Recognizer
def capture_output(pdf_bytes):
    result_data = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = result_data  # Redirect stdout to capture print statements
    
    poller = document_analysis_client.begin_analyze_document("prebuilt-layout", pdf_bytes)
    result = poller.result()

    # Output whether the document contains handwritten content
    for idx, style in enumerate(result.styles):
        content_type = "handwritten" if style.is_handwritten else "no handwritten"
        print(f"Doc contains {content_type} content")

    # Output line contents with original text (no encoding)
    for page in result.pages:
        for line_idx, line in enumerate(page.lines):
            print(f"L{line_idx}: {line.content}")

        # Output selection marks
        for selection_mark in page.selection_marks:
            print(f"Selection: {selection_mark.state}, Confidence: {selection_mark.confidence}")

    # Output table information
    for table_idx, table in enumerate(result.tables):
        print(f"Table {table_idx}: {table.row_count} rows, {table.column_count} cols")
        # Output each cell's content, skipping empty cells
        for cell in table.cells:
            if cell.content.strip():  # Check if the cell content is not empty
                print(f"C[{cell.row_index}][{cell.column_index}]: {cell.content}")

    sys.stdout = old_stdout  # Reset stdout
    return result_data.getvalue()

# Function to send captured output to OpenAI assistant
def send_to_assistant(captured_output):
    thread = openai.beta.threads.create()
    my_thread_id = thread.id

    # Send the captured output as-is
    openai.beta.threads.messages.create(
        thread_id=my_thread_id,
        role="user",
        content=captured_output
    )

    # Run the assistant
    run = openai.beta.threads.runs.create(
        thread_id=my_thread_id,
        assistant_id=assistant_id,
    ) 

    return run.id, thread.id

def check_status(run_id, thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status

# API endpoint to handle PDF file upload and return JSON
@app.route('/process-pdf', methods=['POST'])
async def process_pdf():
    # Meg kell várni a request.files-t
    files = await request.files

    if 'file' not in files:
        return jsonify({"error": "No file uploaded"}), 400

    file = files['file']
    if file.content_type != 'application/pdf':
        return jsonify({"error": "Invalid file type. Please upload a PDF"}), 400

    # Convert the PDF file to bytes and process (no need for await here)
    pdf_bytes = io.BytesIO(file.read())
    captured_output = capture_output(pdf_bytes)

    # Send the captured output to OpenAI assistant
    my_run_id, my_thread_id = send_to_assistant(captured_output)

    # Check the status in a loop until it's completed
    status = check_status(my_run_id, my_thread_id)
    while status != "completed":
        status = check_status(my_run_id, my_thread_id)
        time.sleep(2)

    # Retrieve the assistant's response
    response = openai.beta.threads.messages.list(
        thread_id=my_thread_id
    )

    if response.data:
        # Clean the response by removing ```json at the beginning and backticks at the end
        assistant_response = response.data[0].content[0].text.value
        cleaned_response = assistant_response.replace("```json", "").rstrip("`").strip()

        # Return the cleaned response as JSON
        return jsonify({"assistant_response": cleaned_response})
    else:
        return jsonify({"error": "No response from assistant"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port)
