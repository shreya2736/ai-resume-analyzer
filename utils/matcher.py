import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# MASTER SKILL LIST
# ─────────────────────────────────────────────
SKILLS_LIST = [
    # Programming Languages
    "Python", "Java", "JavaScript", "C", "C++", "C#", "R", "Swift", "Kotlin", "Go",
    "Rust", "TypeScript", "PHP", "Ruby", "Scala", "MATLAB",

    # Web Development
    "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Flask", "Django",
    "FastAPI", "REST API", "GraphQL", "Bootstrap", "Tailwind",

    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "OpenCV",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis",
    "Firebase", "Oracle", "NoSQL",

    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub",
    "CI/CD", "Linux", "Bash", "Terraform", "Jenkins",

    # AI & GenAI
    "Generative AI", "LLM", "Prompt Engineering", "Langchain",
    "Hugging Face", "OpenAI", "Gemini", "RAG",

    # Data Engineering
    "Spark", "Hadoop", "Kafka", "Airflow", "ETL", "Power BI",
    "Tableau", "Excel", "Data Analysis", "Data Visualization",

    # Soft Skills
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Time Management"
]

# ─────────────────────────────────────────────
# Load sentence transformer model once
# Cached at module level so it doesn't
# reload on every function call
# ─────────────────────────────────────────────
_semantic_model = None

def get_semantic_model():
    """
    Load the sentence transformer model.
    Only loads once — reuses on subsequent calls.
    """
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _semantic_model


# ─────────────────────────────────────────────
# SKILL EXTRACTION
# ─────────────────────────────────────────────
def extract_skills(text):
    """
    Extract skills from text by matching
    against SKILLS_LIST using word boundaries.
    """
    found_skills = []
    text_lower   = text.lower()

    for skill in SKILLS_LIST:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────
def calculate_tfidf_score(resume_text, jd_text):
    """
    Score 1 — TF-IDF Score (keyword matching).
    Checks for exact keyword overlap between
    resume and job description.
    Returns a score from 0 to 100.
    """
    vectorizer = TfidfVectorizer(stop_words='english')

    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score        = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        print(f"TF-IDF scoring failed: {e}")
        return 0


def calculate_semantic_score(resume_text, jd_text):
    """
    Semantic score using chunked encoding.
    Splits resume into paragraphs and scores
    each chunk against the JD, taking the average
    of top matches for a fairer score.
    """
    try:
        model = get_semantic_model()

        # Split resume into chunks (paragraphs)
        chunks = [c.strip() for c in resume_text.split('\n') if len(c.strip()) > 30]

        # If no chunks found fall back to full text
        if not chunks:
            chunks = [resume_text]

        # Encode JD and all chunks
        jd_embedding     = model.encode(jd_text, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks,  convert_to_tensor=True)

        # Get similarity score for each chunk vs JD
        scores = util.cos_sim(chunk_embeddings, jd_embedding)

        # Take average of top 5 scoring chunks
        top_scores = sorted(scores.tolist(), reverse=True)[:5]
        avg_score  = sum(s[0] for s in top_scores) / len(top_scores)

        return round(max(0, min(avg_score * 100, 100)), 2)

    except Exception as e:
        print(f"Semantic scoring failed: {e}")
        return 0


def calculate_hybrid_score(resume_text, jd_text):
    """
    Final Score — Hybrid (TF-IDF + Semantic).
    Combines both scoring methods:
      - TF-IDF (40%)  : catches exact keyword matches
      - Semantic (60%): catches conceptual similarity

    Formula: (0.4 × tfidf) + (0.6 × semantic)

    Returns:
        final_score  : the weighted hybrid score
        tfidf_score  : raw TF-IDF score (for display)
        semantic_score: raw semantic score (for display)
    """
    tfidf_score    = calculate_tfidf_score(resume_text, jd_text)
    semantic_score = calculate_semantic_score(resume_text, jd_text)

    # New
    final_score = round((0.5 * tfidf_score) + (0.5 * semantic_score), 2)

    return final_score, tfidf_score, semantic_score


# ─────────────────────────────────────────────
# SKILL COMPARISON
# ─────────────────────────────────────────────
def get_missing_skills(resume_skills, jd_skills):
    """
    Skills in JD but missing from resume.
    """
    resume_set = set(s.lower() for s in resume_skills)
    return [s for s in jd_skills if s.lower() not in resume_set]


def get_common_skills(resume_skills, jd_skills):
    """
    Skills present in both resume and JD.
    """
    resume_set = set(s.lower() for s in resume_skills)
    return [s for s in jd_skills if s.lower() in resume_set]