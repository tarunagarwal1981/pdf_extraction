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

def chunk_text(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into smaller chunks, ensuring we stay within token limits
    
    Args:
        text: The input text to chunk
        max_chunk_size: Maximum characters per chunk (default 2000 to stay well within token limits)
    """
    # First split by error code entries (ACUXX_)
    entries = []
    current_entry = ""
    
    for line in text.split('\n'):
        if line.strip().startswith('ACUXX_'):
            if current_entry:
                entries.append(current_entry.strip())
            current_entry = line
        else:
            current_entry += '\n' + line
    
    if current_entry:
        entries.append(current_entry.strip())
    
    # Now group entries into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    for entry in entries:
        entry_size = len(entry)
        if current_size + entry_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [entry]
            current_size = entry_size
        else:
            current_chunk.append(entry)
            current_size += entry_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def process_chunk_with_llm(chunk: str, client: OpenAI) -> List[Dict]:
    """Process text chunk with GPT-3.5-turbo and extract structured data"""
    system_prompt = """You are a technical documentation parser. Extract error codes and their details from the input text.
    
    Example Input Format:
    ACUXX_009904
    Ch35,0099,Prop. Valve Test Set Poin->Suprv...
    Description: MBD Special test purposes only
    Cause: MBD special test equipment not connected
    Effect: No effect on engine...
    Sugg. Action: No action required...
    
    Return ONLY a JSON array with this EXACT structure:
    [
        {
            "heading": "full error code and title",
            "description": "content after Description:",
            "cause": "content after Cause:",
            "effect": "content after Effect:",
            "sugg_action": "content after Sugg. Action:"
        }
    ]"""

    try:
        # Add chunk size debugging
        with st.expander("Debug Information", expanded=False):
            st.text(f"Processing chunk of size: {len(chunk)} characters")
            st.text("First 500 characters of chunk:")
            st.code(chunk[:500])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract data from:\n\n{chunk}"}
            ],
            temperature=0,
            max_tokens=1000,
            presence_penalty=0,
            frequency_penalty=0
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            parsed_data = json.loads(content)
            # Validate required fields
            validated_data = []
            for entry in parsed_data:
                if all(key in entry and entry[key] for key in ["heading", "description", "cause", "effect", "sugg_action"]):
                    validated_data.append(entry)
            return validated_data
        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON in response: {str(je)}")
            st.text("Raw response was:")
            st.code(content)
            return []
            
    except Exception as e:
        st.error(f"API Error: {str(e)}")
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
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_data = process_chunk_with_llm(chunk, client)
                all_data.extend(chunk_data)
                progress_bar.progress((i + 1) / len(chunks))
            
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
