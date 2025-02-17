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
    system_prompt = """You must extract technical documentation data and return it in valid JSON format.
    Extract ONLY these fields and return them in an array:
    - heading (including full error code and title)
    - description (full description text)
    - cause (full cause text)
    - effect (full effect text)
    - sugg_action (full suggested action text)

    Return the data in this exact format:
    [
        {
            "heading": "ACUXX_009904 Ch35,0099,Prop. Valve Test Set Poin->Suprv",
            "description": "MBD Special test purposes only",
            "cause": "MBD special test equipment not connected",
            "effect": "No effect on engine",
            "sugg_action": "No action required"
        }
    ]

    Important: Ensure the response is valid JSON. Do not include any text before or after the JSON array.
    Escape any special characters in the text properly."""

    # Split text into smaller chunks
    max_chunk_length = 8000  # Reduced chunk size
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    all_results = []
    try:
        for chunk in chunks:
            if not chunk.strip():  # Skip empty chunks
                continue

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract and format as JSON: {chunk}"}
                ],
                temperature=0,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )

            content = response.choices[0].message.content

            if debug_mode:
                st.write(f"Raw API Response: {content[:500]}...")

            try:
                # Clean the response
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                # Ensure content is wrapped in array brackets
                if not content.startswith('['):
                    content = f'[{content}]'
                if not content.endswith(']'):
                    content = f'{content}]'

                # Parse JSON
                results = json.loads(content)

                # Validate results structure
                if isinstance(results, list):
                    for item in results:
                        if all(k in item for k in ['heading', 'description', 'cause', 'effect', 'sugg_action']):
                            all_results.append(item)
                        else:
                            if debug_mode:
                                st.warning(f"Skipping invalid entry: {item}")
                else:
                    if debug_mode:
                        st.warning("Received non-list JSON response")

            except json.JSONDecodeError as e:
                if debug_mode:
                    st.error(f"Error parsing OpenAI JSON response: {str(e)}")
                    st.write("Problematic content:", content)
                continue

        return all_results
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return []

def process_with_anthropic(text, client):
    """Process text with Anthropic API"""
    system_prompt = """Extract technical documentation data and return it in valid JSON format.
    Extract ONLY these fields and return them in an array:
    - heading (including full error code and title)
    - description (full description text)
    - cause (full cause text)
    - effect (full effect text)
    - sugg_action (full suggested action text)

    Return the data in this exact format with no additional text."""

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        try:
            content = response.content[0].text
            return json.loads(content)
        except json.JSONDecodeError as e:
            if debug_mode:
                st.error(f"Error parsing Anthropic JSON response: {str(e)}")
            return []
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
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Initialize API clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)

            # Preprocess and split into entries
            entries = preprocess_text(text)

            # Process entries
            all_data = []
            progress_bar = st.progress(0)
            status_container = st.empty()

            for i, entry in enumerate(entries):
                status_container.text(f"Processing entry {i+1} of {len(entries)}...")

                try:
                    if api_choice == "OpenAI GPT-3.5":
                        chunk_data = process_with_openai(entry, openai_client)
                    else:
                        chunk_data = process_with_anthropic(entry, anthropic_client)

                    if isinstance(chunk_data, list) and chunk_data:
                        all_data.extend(chunk_data)
                        if debug_mode:
                            status_container.text(f"Successfully processed entry {i+1}")
                    else:
                        if debug_mode:
                            status_container.warning(f"No valid data extracted from entry {i+1}")

                except Exception as e:
                    st.error(f"Error processing entry {i+1}: {str(e)}")
                    continue

                progress_bar.progress((i + 1) / len(entries))

            status_container.empty()

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

# Debug Information
if debug_mode and 'entries' in locals():
    st.subheader("Debug Information")
    st.write(f"Number of entries detected: {len(entries)}")
    st.write("First entry preview:")
    st.code(entries[0] if entries else "No entries found")
