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
# SENTENCE TRANSFORMER MODEL
# Loaded once and cached for reuse
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
    Returns a list of matched skills.
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
    Score 2 — Semantic Score (meaning matching).
    Splits resume into chunks and scores each
    chunk against JD. Takes average of top matches.
    Returns a score from 0 to 100.
    """
    try:
        model = get_semantic_model()

        # Split resume into meaningful chunks
        chunks = [c.strip() for c in resume_text.split('\n') if len(c.strip()) > 30]

        # Fall back to full text if no chunks found
        if not chunks:
            chunks = [resume_text]

        # Encode JD and all resume chunks
        jd_embedding     = model.encode(jd_text, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks,  convert_to_tensor=True)

        # Score each chunk against the JD
        scores = util.cos_sim(chunk_embeddings, jd_embedding)

        # Average of top 5 matching chunks
        top_scores = sorted(scores.tolist(), reverse=True)[:5]
        avg_score  = sum(s[0] for s in top_scores) / len(top_scores)

        return round(max(0, min(avg_score * 100, 100)), 2)

    except Exception as e:
        print(f"Semantic scoring failed: {e}")
        return 0


def calculate_skill_score(resume_text, jd_text):
    """
    Score 3 — Skill Overlap Score.
    Calculates what percentage of JD skills
    are present in the resume.

    This is the most intuitive score:
    - If JD needs 5 skills and resume has 4 → 80%
    - If JD needs 5 skills and resume has 2 → 40%

    Returns a score from 0 to 100.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    # If JD has no detectable skills return 0
    if not jd_skills:
        return 0

    # Count how many JD skills appear in resume
    resume_skill_set = set(s.lower() for s in resume_skills)
    matched_count    = sum(1 for s in jd_skills if s.lower() in resume_skill_set)

    score = (matched_count / len(jd_skills)) * 100
    return round(score, 2)


def calculate_hybrid_score(resume_text, jd_text):
    """
    Final Score — Three-way Hybrid.

    Combines three scoring methods:
      - Skill Overlap (40%) : % of JD skills found in resume
      - TF-IDF       (30%) : exact keyword matching
      - Semantic     (30%) : conceptual similarity

    Skill overlap gets highest weight because it's
    the most direct and meaningful ATS metric.

    Formula:
        (0.4 × skill_score) + (0.3 × tfidf) + (0.3 × semantic)

    Returns:
        final_score   : weighted hybrid score (0-100)
        tfidf_score   : raw TF-IDF score (for display)
        semantic_score: raw semantic score (for display)
        skill_score   : raw skill overlap score (for display)
    """
    tfidf_score    = calculate_tfidf_score(resume_text, jd_text)
    semantic_score = calculate_semantic_score(resume_text, jd_text)
    skill_score    = calculate_skill_score(resume_text, jd_text)

    final_score = round(
        (0.4 * skill_score) +
        (0.3 * tfidf_score) +
        (0.3 * semantic_score),
        2
    )

    return final_score, tfidf_score, semantic_score, skill_score


# ─────────────────────────────────────────────
# SKILL COMPARISON
# ─────────────────────────────────────────────
def get_missing_skills(resume_skills, jd_skills):
    """
    Skills present in JD but missing from resume.
    """
    resume_set = set(s.lower() for s in resume_skills)
    return [s for s in jd_skills if s.lower() not in resume_set]


def get_common_skills(resume_skills, jd_skills):
    """
    Skills present in both resume and JD.
    """
    resume_set = set(s.lower() for s in resume_skills)
    return [s for s in jd_skills if s.lower() in resume_set]