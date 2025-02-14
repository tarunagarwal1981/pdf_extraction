import streamlit as st
import PyPDF2
import pandas as pd
import io
from openai import OpenAI
import json
from typing import List, Dict

st.title("Technical Documentation Data Extractor")

# Initialize OpenAI client with secret
client = OpenAI(api_key=st.secrets["api_key"])

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_text(text: str) -> str:
    """Clean and preprocess the text"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Replace multiple newlines with single newline
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    return text

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into very small chunks, processing one or few entries at a time"""
    # Clean the text first
    text = clean_text(text)
    
    # Split by ACUXX_ entries
    chunks = []
    current_entry = ""
    
    for line in text.split('\n'):
        if line.startswith('ACUXX_'):
            if current_entry:
                # Only add if it looks like a complete entry
                if all(marker in current_entry.lower() for marker in ['description:', 'cause:', 'effect:', 'sugg']):
                    chunks.append(current_entry.strip())
            current_entry = line
        else:
            current_entry += '\n' + line
    
    # Add the last entry if it exists
    if current_entry and all(marker in current_entry.lower() for marker in ['description:', 'cause:', 'effect:', 'sugg']):
        chunks.append(current_entry.strip())
    
    # Debug information
    st.write(f"Found {len(chunks)} potential entries to process")
    
    return chunks

def process_chunk_with_llm(chunk: str, client: OpenAI) -> List[Dict]:
    """Process text chunk with GPT-3.5-turbo and extract structured data"""
    # Shorter prompt to save tokens
    system_prompt = """Extract technical documentation data into JSON. Format:
    [{
        "heading": "error code and title",
        "description": "description content",
        "cause": "cause content",
        "effect": "effect content",
        "sugg_action": "suggested action content"
    }]"""

    try:
        # Debug the chunk size
        chunk_size = len(chunk)
        if chunk_size > 1000:  # Warning if chunk is too large
            st.warning(f"Large chunk detected ({chunk_size} chars). Processing may fail.")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Show processing details in expander
        with st.expander(f"Processing details for chunk starting with: {chunk[:50]}...", expanded=False):
            st.text(f"Chunk size: {chunk_size} characters")
            st.text("Input chunk:")
            st.code(chunk)
            st.text("Raw response:")
            st.code(content)
        
        try:
            parsed_data = json.loads(content)
            validated_data = []
            for entry in parsed_data:
                if all(key in entry and entry[key] for key in ["heading", "description", "cause", "effect", "sugg_action"]):
                    validated_data.append(entry)
            return validated_data
        except json.JSONDecodeError as je:
            st.error(f"JSON parsing error: {str(je)}")
            return []
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return []

def save_to_excel(data: List[Dict]) -> bytes:
    """Convert extracted data to Excel file"""
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Extracted Data', index=False)
    return output.getvalue()

def save_to_csv(data: List[Dict]) -> bytes:
    """Convert extracted data to CSV file"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file:
    # Process button
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            # Chunk the text
            chunks = chunk_text(text)
            
            # Process each chunk and combine results
            all_data = []
            total_chunks = len(chunks)
            
            st.write(f"Processing {total_chunks} entries...")
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_data = process_chunk_with_llm(chunk, client)
                if chunk_data:
                    all_data.extend(chunk_data)
                progress_bar.progress((i + 1) / total_chunks)
                
                # Add a small delay between API calls if needed
                if i < total_chunks - 1:
                    time.sleep(0.5)  # 500ms delay between calls
            
            # Show preview of extracted data
            if all_data:
                st.success(f"Successfully extracted {len(all_data)} entries!")
                
                st.subheader("Preview of extracted data:")
                preview_df = pd.DataFrame(all_data)
                st.dataframe(preview_df)
                
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
                st.warning("No valid entries were extracted. Please check the debug information.")
else:
    st.info("Please upload a PDF file to begin.")
