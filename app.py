import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import json
import os
from google import genai
from google.genai import types

# ----------------------------------------------------
# 1. Page Configuration & Title
# ----------------------------------------------------
st.set_page_config(
    page_title="Fact-Check Agent | CogCulture Assessment",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Automated AI Fact-Checking Agent")
st.caption("Upload marketing briefs, case studies, or PDFs to verify statistics and claims against live web data in real-time.")

# ----------------------------------------------------
# 2. Authentication & Sidebar Setup
# ----------------------------------------------------
st.sidebar.header("Configuration")
# Allow user to input API key or fall back to Streamlit secrets environment variable
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get a key from Google AI Studio")

# Resolve API key presence
final_api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY")

if not final_api_key:
    st.sidebar.warning("⚠️ Please provide a Gemini API Key to proceed.")
    st.info("💡 To test this app, obtain an API key from Google AI Studio and input it in the sidebar.")

# ----------------------------------------------------
# 3. Helper Functions: PDF Extraction & AI Analysis
# ----------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def process_fact_check(document_text, api_key_input):
    """Uses Gemini 2.5 Flash with Google Search Grounding to parse and verify facts."""
    # Initialize the official Google GenAI client
    client = genai.Client(api_key=api_key_input)
    
    # Construct a bulletproof system prompt requiring JSON output matching evaluation schema
    prompt = f"""
    You are an elite Fact-Checking Verification Agent. Your task is to analyze the text provided below, identify any specific quantitative data, figures, statistics, financial data, or concrete technical claims, and verify them against the live internet using search grounding.

    Document Text to Analyze:
    \"\"\"{document_text}\"\"\"

    Instructions:
    1. Extract all key verifiable statistics, dates, or factual figures.
    2. Use your search grounding capabilities to look up the actual, correct current data on the live web.
    3. Categorize each claim into one of these strict classifications:
       - Verified: The claim matches current live data perfectly.
       - Inaccurate: The claim is partially true but uses outdated statistics or has minor errors.
       - False: The claim is completely wrong or no evidence online supports it.
    4. Provide the correct real fact/statistic alongside the live search evidence you found.

    Your output MUST be a valid JSON array of objects. Do not wrap it in markdown code blocks. Just output raw JSON text matching this schema:
    [
      {{
        "claim": "The exact statement or stat extracted from the PDF",
        "status": "Verified" or "Inaccurate" or "False",
        "extracted_stat": "The specific value/metric stated in the document",
        "real_fact": "The actual current metric or true state discovered via live web search",
        "source_evidence": "Brief description of the source verification text or online consensus"
      }}
    ]
    """
    
    try:
        # Request generation with search grounding turned on natively
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                # Enable live Google Search grounding tool
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1 # Low temperature for analytical consistency
                
            )
        )
        
        # Parse output safely
        result_json = json.loads(response.text)
        return result_json
    except json.JSONDecodeError:
        # Fallback if markdown wraps or format slips
        try:
            clean_text = response.text.strip().strip("```json").strip("```")
            return json.loads(clean_text)
        except Exception:
            st.error("Failed to parse the structured model output.")
            return None
    except Exception as e:
        st.error(f"API Processing error: {e}")
        return None

# ----------------------------------------------------
# 4. Core App UI and Flow Logic
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Text extraction block
    with st.spinner("Extracting content from PDF document..."):
        document_content = extract_text_from_pdf(uploaded_file)
        
    if document_content:
        with st.expander("View Extracted Raw Text"):
            st.text(document_content[:2000] + ("..." if len(document_content) > 2000 else ""))
            
        # Trigger Verification Action
        if st.button("Run Automated Fact-Check", type="primary"):
            if not final_api_key:
                st.error("Cannot proceed: Missing API key. Please input your key in the sidebar.")
            else:
                with st.spinner("Analyzing claims and cross-referencing live web data... This takes a few seconds."):
                    report_data = process_fact_check(document_content, final_api_key)
                    
                if report_data:
                    st.subheader("📊 Audit & Verification Report")
                    
                    # Convert json data into a clean pandas dataframe
                    df = pd.DataFrame(report_data)
                    
                    # Custom coloring mapping function for visual validation highlight
                    def highlight_status(val):
                        if val == 'Verified':
                            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        elif val == 'Inaccurate':
                            return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                        elif val == 'False':
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                        return ''
                    
                    # Render styled interactive data table
                    try:
                        styled_df = df.style.applymap(highlight_status, subset=['status'])
                        st.dataframe(styled_df, use_container_width=True)
                    except Exception:
                        st.dataframe(df, use_container_width=True)
                        
                    # Visual summary metrics
                    st.markdown("---")
                    st.subheader("Summary Breakdown")
                    status_counts = df['status'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Verified Claims", status_counts.get("Verified", 0))
                    col2.metric("Inaccurate Claims", status_counts.get("Inaccurate", 0))
                    col3.metric("False Claims", status_counts.get("False", 0))
