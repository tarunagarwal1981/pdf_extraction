import streamlit as st
import PyPDF2
import pandas as pd
import io
from openai import OpenAI
import json
import time
import re
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

def extract_entries(text: str) -> List[str]:
    """Extract individual ACUXX entries from text using regex"""
    # Split text at ACUXX_ patterns
    entries = re.split(r'(?=ACUXX_\d{6})', text)
    
    # Filter out any entries that don't start with ACUXX_
    entries = [entry.strip() for entry in entries if entry.strip().startswith('ACUXX_')]
    
    # Debug information
    st.write(f"Found {len(entries)} potential entries")
    return entries

def clean_entry(entry: str) -> str:
    """Clean and format a single entry"""
    # Remove excessive whitespace and newlines
    entry = re.sub(r'\s+', ' ', entry)
    
    # Add newlines before main sections for better formatting
    sections = ['Description:', 'Cause:', 'Effect:', 'Sugg. Action:']
    for section in sections:
        entry = re.sub(f'({section})', r'\n\1', entry)
    
    return entry.strip()

def process_single_entry(entry: str, client: OpenAI) -> List[Dict]:
    """Process a single ACUXX entry"""
    # Skip if entry is too long
    if len(entry) > 2000:
        st.warning(f"Entry too long ({len(entry)} chars), skipping...")
        return []
        
    system_prompt = """Extract technical data into this exact JSON format:
    [{
        "heading": "error code and full title",
        "description": "text after Description:",
        "cause": "text after Cause:",
        "effect": "text after Effect:",
        "sugg_action": "text after Sugg. Action:"
    }]"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry}
            ],
            temperature=0,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Show processing details in expander
        with st.expander(f"Entry details: {entry[:50]}...", expanded=False):
            st.text("Input entry:")
            st.code(entry)
            st.text("Raw response:")
            st.code(content)
        
        try:
            parsed_data = json.loads(content)
            validated_data = []
            for item in parsed_data:
                if all(key in item and item[key] for key in ["heading", "description", "cause", "effect", "sugg_action"]):
                    validated_data.append(item)
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
            
            # Extract and clean individual entries
            entries = extract_entries(text)
            entries = [clean_entry(entry) for entry in entries]
            
            # Process entries
            all_data = []
            total_entries = len(entries)
            
            st.write(f"Processing {total_entries} entries...")
            progress_bar = st.progress(0)
            
            for i, entry in enumerate(entries):
                # Skip empty or invalid entries
                if not entry or not entry.startswith('ACUXX_'):
                    continue
                    
                entry_data = process_single_entry(entry, client)
                if entry_data:
                    all_data.extend(entry_data)
                progress_bar.progress((i + 1) / total_entries)
                
                # Add a small delay between API calls
                if i < total_entries - 1:
                    time.sleep(0.5)
            
            # Show results
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
