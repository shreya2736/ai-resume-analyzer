# AI Resume Analyzer 📄

An intelligent Streamlit web application that analyzes resumes against job descriptions. It provides ATS-style matching scores, skill gap analysis, AI-powered feedback, role recommendations, and a project rewriter – all powered by Groq’s Llama 3.3 70B model.


##  Features

- **PDF Resume Parsing** – Extracts text from PDFs using `pdfplumber` (fallback to `PyPDF2`).
- **Skill Extraction** – Matches technical & soft skills from a curated master list.
- **Hybrid ATS Scoring** — Four-way scoring system:
  - **Skill Overlap (50%)** — percentage of JD skills found in the resume.
  - **Title Match (20%)** — job role alignment using a curated job titles list.
  - **Semantic (20%)** — conceptual similarity using sentence transformers.
  - **TF-IDF (10%)** — exact keyword matching.
- **Skill Gap Analysis** – Shows matched and missing skills.
- **AI Feedback** – Direct, actionable suggestions to improve your resume.
- **Shared API Key** — Works out of the box for all users. Optionally paste your own key for unlimited use.
- **Role Recommendations** – Top 5 job titles with match % and upskill tips.
- **Score Breakdown Tab** — Shows all four sub-scores (Skill Overlap, Title Match, Semantic, TF-IDF) with individual progress bars and explanations.
- **Resume Project Rewriter** – Turns weak bullet points into strong, ATS-friendly descriptions.
- **Downloadable Report** – Generates a plain‑text summary of the entire analysis.
- **Custom CSS** – Modern dark UI with badges, progress bars, and tabs.

##  Tech Stack

- **Frontend/App**: Streamlit
- **LLM API**: Groq (Llama 3.3 70B)
- **Semantic Scoring**: Sentence‑Transformers (`all-MiniLM-L6-v2`)
- **Keyword Scoring**: Scikit‑learn (TF‑IDF + cosine similarity)
- **Skill Scoring** : Regex-based skill overlap
- **Title Matching** : Lightweight job title extraction (alternative to full NER)
- **PDF Parsing**: pdfplumber (primary) , PyPDF2 (fallback)

## Project Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – Python dependencies
- `.gitignore` - Excludes secrets from GitHub
- `.streamlit/`
  - `secrets.toml ` - Groq API key (not pushed to GitHub)
- `utils/`
  - `__init__.py`
  - `parser.py` – PDF text extraction
  - `matcher.py` – Skill extraction, job title matching, TF-IDF, semantic scoring, hybrid scoring
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

##  Running the App
    ```bash
    streamlit run app.py
The app will open in your default browser at http://localhost:8501.

##  Getting a Groq API Key

1. The AI features (feedback, role recommendations, rewriting, report generation) require a Groq API key.

2. Option A — Use the shared key (no setup needed)
The app comes with a shared API key built in. Just open the app and use it directly.

3. Option B — Use your own key (recommended for unlimited use)

  - Go to console.groq.com
  - Sign up — no credit card required
  - Navigate to API Keys → Create API Key
  - Paste it in the sidebar of the app
Your key takes priority over the shared key when entered.

## How to Use

1. Open the app (no setup needed — shared key works out of the box)
2. Upload your resume in PDF format
3. Paste a job description into the text area
4. Click Analyze My Resume
5. Explore results across five tabs:
  -  ATS Score — hybrid match score, matched and missing skills
  -  AI Feedback — specific section improvements and skills to learn
  -  Role Recommendations — suggested job titles with match % and     upskill tips
  -  All Skills — side-by-side skill lists from resume and JD
  -  Score Breakdown — individual sub-scores with progress bars and explanation
6. Scroll down to Download Report — generate and download a full analysis as .txt
7. Scroll down to Resume Project Rewriter — paste a weak description and get a polished version

##  Configuration

- Skill List – Edit SKILLS_LIST in matcher.py to add or remove skills.
- Scoring Weights – In calculate_hybrid_score(), the default formula is **(0.5 × Skill Overlap) + (0.2 × Title Match) + (0.2 × Semantic) + (0.1 × TF-IDF)**. Adjust as needed.
- Job Titles List – Edit `JOB_TITLES` in `matcher.py` to add or remove job titles for title matching.
- LLM Model – Change model="llama-3.3-70b-versatile" in _call_groq() to another Groq model.

##  Live Demo

Try the app here: [AI Resume Analyzer](https://ai-resume-analyzer-djvcztykawa4bkpqmkuvue.streamlit.app/)
