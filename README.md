# AI Fact-Checking Agent - CogCulture Technical Assessment

An automated, deployment-ready production application that parses PDF uploads, extracts data/statistical insights, and leverages search-grounded LLM verification frameworks to act as an un-hallucinated "Truth Layer".

## Features
- **Dynamic File Processing**: Instant on-the-fly parsing of multi-page PDFs using PyPDF2.
- **Real-Time Grounding Engine**: Natively calls Google Search index connections via the GenAI standard framework to bypass stale knowledge limitations.
- **Fraud/Trap Detection**: Explicitly engineered prompts categorize insights into `Verified`, `Inaccurate`, or `False` datasets to flag deliberate traps.

## How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit command: `streamlit run app.py`
