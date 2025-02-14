import streamlit as st
import PyPDF2
import pandas as pd
import io
from openai import OpenAI
import json

st.title("Technical Documentation Data Extractor")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=4000):
    """Split text into chunks of approximately chunk_size characters"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_chunk_with_llm(chunk, client):
    """Process text chunk with GPT-3.5-turbo and extract structured data"""
    # Clean and prepare the text
    chunk = chunk.replace('\n', ' ').replace('\r', ' ')
    while '  ' in chunk:
        chunk = chunk.replace('  ', ' ')
    system_prompt = """You are a technical documentation parser. Extract data from the given text into JSON format.
    Rules:
    1. Look for entries that follow the pattern: error code/title followed by Description, Cause, Effect, and Suggested Action
    2. Each entry should have all fields (heading, description, cause, effect, sugg_action)
    3. Return the data in this exact JSON structure:
    [{
        "heading": "error code and title (e.g., ACUXX_009904)",
        "description": "full description text",
        "cause": "full cause text",
        "effect": "full effect text",
        "sugg_action": "full suggested action text"
    }]
    4. Ensure the JSON is properly formatted and all text values are properly escaped
    5. Skip any incomplete entries where all fields are not present
    6. Do not add any explanatory text - return only the JSON array"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract structured data from this text: {chunk}"}
            ],
            temperature=0,
            max_tokens=2000,
            presence_penalty=0,
            frequency_penalty=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error processing chunk: {str(e)}")
        return []

def save_to_excel(data):
    """Convert extracted data to Excel file"""
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Extracted Data', index=False)
    return output.getvalue()

def save_to_csv(data):
    """Convert extracted data to CSV file"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# API key input
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file and api_key:
    client = OpenAI(api_key=api_key)
    
    # Process button
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            # Chunk the text
            chunks = chunk_text(text)
            
            # Process each chunk and combine results
            all_data = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_data = process_chunk_with_llm(chunk, client)
                all_data.extend(chunk_data)
                progress_bar.progress((i + 1) / len(chunks))
            
            # Show preview of extracted data
            st.write("Preview of extracted data:")
            preview_df = pd.DataFrame(all_data)
            st.dataframe(preview_df.head())
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                excel_data = save_to_excel(all_data)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="extracted_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                csv_data = save_to_csv(all_data)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="extracted_data.csv",
                    mime="text/csv"
                )
else:
    st.info("Please upload a PDF file and enter your OpenAI API key to begin.")