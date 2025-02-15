import streamlit as st
import PyPDF2
import pandas as pd
import io
from openai import OpenAI
from anthropic import Anthropic
import json
import re

st.set_page_config(page_title="Technical Documentation Extractor", layout="wide")
st.title("Technical Documentation Data Extractor")

# Configure API settings from secrets
if 'OPENAI_API_KEY' not in st.secrets or 'ANTHROPIC_API_KEY' not in st.secrets:
    st.error("Please set OPENAI_API_KEY and ANTHROPIC_API_KEY in your Streamlit secrets.")
    st.stop()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Clean and structure the text for better extraction"""
    # Replace multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Identify entry boundaries (assuming entries start with ACUXX_)
    entries = re.split(r'(?=ACUXX_\d+)', text)
    
    # Remove empty entries
    entries = [e.strip() for e in entries if e.strip()]
    
    return entries

def process_with_openai(text, client):
    """Process text with OpenAI API"""
    system_prompt = """Extract technical documentation data from the text below. Return ONLY a JSON array containing entries with exactly these fields:
    - heading (including full error code and title)
    - description (full description text)
    - cause (full cause text)
    - effect (full effect text)
    - sugg_action (full suggested action text)

    Example format:
    [
        {
            "heading": "ACUXX_009904 Ch35,0099,Prop. Valve Test Set Poin->Suprv",
            "description": "MBD Special test purposes only",
            "cause": "MBD special test equipment not connected",
            "effect": "No effect on engine",
            "sugg_action": "No action required"
        }
    ]

    Return ONLY the JSON array with no additional text. Ensure all entries have all required fields."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return []

def process_with_anthropic(text, client):
    """Process text with Anthropic API"""
    system_prompt = """Extract technical documentation data from the text below. Return ONLY a JSON array containing entries with exactly these fields:
    - heading (including full error code and title)
    - description (full description text)
    - cause (full cause text)
    - effect (full effect text)
    - sugg_action (full suggested action text)

    Return the data in this exact format with no additional text:
    [
        {
            "heading": "ACUXX_009904 Ch35,0099,Prop. Valve Test Set Poin->Suprv",
            "description": "MBD Special test purposes only",
            "cause": "MBD special test equipment not connected",
            "effect": "No effect on engine",
            "sugg_action": "No action required"
        }
    ]"""

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        content = response.content[0].text
        return json.loads(content)
    except Exception as e:
        st.error(f"Anthropic API Error: {str(e)}")
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

# Sidebar for settings
st.sidebar.title("Settings")
api_choice = st.sidebar.radio(
    "Select API Provider",
    ["OpenAI GPT-3.5", "Anthropic Claude"]
)

# Initialize API clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file:
    # Process button
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            # Preprocess and split into entries
            entries = preprocess_text(text)
            
            # Process entries
            all_data = []
            progress_bar = st.progress(0)
            
            for i, entry in enumerate(entries):
                if api_choice == "OpenAI GPT-3.5":
                    chunk_data = process_with_openai(entry, openai_client)
                else:
                    chunk_data = process_with_anthropic(entry, anthropic_client)
                
                all_data.extend(chunk_data)
                progress_bar.progress((i + 1) / len(entries))
            
            # Show preview of extracted data
            st.subheader("Preview of extracted data:")
            if all_data:
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
                st.warning("No data was extracted. Please check the PDF format and try again.")
else:
    st.info("Please upload a PDF file to begin.")

# Add section for debugging information
if st.checkbox("Show Debug Information"):
    st.subheader("Debug Information")
    if 'entries' in locals():
        st.write(f"Number of entries detected: {len(entries)}")
        st.write("First entry preview:")
        st.code(entries[0] if entries else "No entries found")
