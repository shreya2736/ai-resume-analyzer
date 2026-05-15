# AI Resume Analyzer 📄

An intelligent Streamlit web application that analyzes resumes against job descriptions. It provides ATS-style matching scores, skill gap analysis, AI-powered feedback, role recommendations, and a project rewriter – all powered by Groq’s Llama 3.3 70B model.


##  Features

- **PDF Resume Parsing** – Extracts text from PDFs using `pdfplumber` (fallback to `PyPDF2`).
- **Skill Extraction** – Matches technical & soft skills from a curated master list.
- **Hybrid ATS Scoring**:
  - **TF‑IDF** (50%) – exact keyword matching.
  - **Semantic** (50%) – conceptual similarity using sentence transformers.
- **Skill Gap Analysis** – Shows matched and missing skills.
- **AI Feedback** – Direct, actionable suggestions to improve your resume.
- **Role Recommendations** – Top 5 job titles with match % and upskill tips.
- **Resume Project Rewriter** – Turns weak bullet points into strong, ATS-friendly descriptions.
- **Downloadable Report** – Generates a plain‑text summary of the entire analysis.
- **Custom CSS** – Modern dark UI with badges, progress bars, and tabs.

##  Tech Stack

- **Frontend/App**: Streamlit
- **LLM API**: Groq (Llama 3.3 70B)
- **NLP/Embeddings**: Sentence‑Transformers (`all-MiniLM-L6-v2`)
- **Text Similarity**: Scikit‑learn (TF‑IDF + cosine similarity)
- **PDF Parsing**: pdfplumber (primary) , PyPDF2 (fallback)

## Project Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – Python dependencies
- `utils/`
  - `parser.py` – PDF text extraction
  - `matcher.py` – Skills extraction, TF‑IDF, semantic scoring
  - `groq_api.py` – Groq API client & helper functions
  - `prompts.py` – Prompt templates for the LLM

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-resume-analyzer.git
   cd ai-resume-analyzer
    
2. Install dependencies

    ```bash
    pip install -r requirements.txt
    sentence-transformers will download the all-MiniLM-L6-v2 model on first use (~80 MB).

##  Getting a Groq API Key
The AI features (feedback, role recommendations, rewriting, report generation) require a Groq API key.
1. Go to console.groq.com
2. Sign up (no credit card required)
3. Navigate to API Keys → Create API Key
4. Copy the key

##  How to Use
1. Enter your Groq API key in the sidebar (required for AI features).
2. Upload your resume (PDF format).
3. Paste a job description into the text area.
4. Click Analyze My Resume.
5. Explore the results across five tabs: ATS Score – hybrid match score, matched/missing skills.
6. AI Feedback – specific improvements, skills to learn, action items.
7. Role Recommendations – suggested job titles based on your resume.
8. All Skills – side‑by‑side skill lists.
9. Score Breakdown – TF‑IDF vs semantic scores.
10. Generate a downloadable report (requires API key).
11. Use the Project Rewriter – paste a weak project description and get a polished version.

##  Configuration

- Skill List – Edit SKILLS_LIST in matcher.py to add or remove skills.
- Scoring Weights – In calculate_hybrid_score(), the default is 50% TF‑IDF + 50% semantic. Adjust as needed.
- LLM Model – Change model="llama-3.3-70b-versatile" in _call_groq() to another Groq model.

