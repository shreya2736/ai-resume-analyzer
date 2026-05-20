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
    "Next.js", "Express.js", "Vercel", "Netlify",

    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "OpenCV",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "XGBoost", "LightGBM", "Random Forest", "Regression",
    "Classification", "Clustering", "Feature Engineering",
    "Data Wrangling", "Model Deployment", "Statistics", "Probability",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis",
    "Firebase", "Oracle", "NoSQL", "MongoDB Atlas",

    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub",
    "CI/CD", "Linux", "Bash", "Terraform", "Jenkins",
    "Streamlit Cloud", "Heroku", "Railway", "Render", "Lambda",
    "Shell Scripting",

    # Tools & Practices
    "Agile", "Scrum", "JIRA", "Confluence", "VS Code", "PyCharm",
    "Postman", "Jupyter", "Streamlit", "Unit Testing", "pytest",

    # AI & GenAI
    "Generative AI", "LLM", "Prompt Engineering", "Langchain",
    "Hugging Face", "OpenAI", "Gemini", "RAG",
    "Fine-tuning", "Vector Database", "Embeddings",
    "Stable Diffusion", "Whisper", "BERT", "Transformers",

    # Data Engineering
    "Spark", "Hadoop", "Kafka", "Airflow", "ETL", "Power BI",
    "Tableau", "Excel", "Data Analysis", "Data Visualization",

    # Soft Skills
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Time Management", "Analytical Thinking",
    "Attention to Detail", "Collaboration", "Presentation",
    "Documentation"
]

# ─────────────────────────────────────────────
# JOB TITLES LIST
# Used for simple title matching —
# a lightweight alternative to full NER
# ─────────────────────────────────────────────
JOB_TITLES = [
    # Software Engineering
    "Software Engineer", "Software Developer", "Full Stack Developer",
    "Frontend Developer", "Backend Developer", "Web Developer",
    "Mobile Developer", "iOS Developer", "Android Developer",
    "Embedded Engineer", "Systems Engineer",

    # Data & ML
    "Data Scientist", "Data Analyst", "Data Engineer",
    "Machine Learning Engineer", "ML Engineer", "AI Engineer",
    "NLP Engineer", "Computer Vision Engineer", "Research Engineer",
    "Research Scientist", "Applied Scientist",

    # Cloud & DevOps
    "DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer",
    "Platform Engineer", "Infrastructure Engineer",

    # Specialized
    "Python Developer", "Java Developer", "JavaScript Developer",
    "React Developer", "Django Developer", "Flask Developer",
    "Database Administrator", "Security Engineer",
    "Blockchain Developer", "Game Developer",

    # Analytics & BI
    "Business Analyst", "Product Analyst", "BI Developer",
    "Analytics Engineer", "Reporting Analyst",

    # Management
    "Tech Lead", "Engineering Manager", "Product Manager",
    "Project Manager", "Scrum Master",

    # Internship / Entry level
    "Software Intern", "Data Science Intern", "ML Intern",
    "Developer Intern", "Engineering Intern"
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
# JOB TITLE EXTRACTION
# ─────────────────────────────────────────────
def extract_job_titles(text):
    """
    Extract job titles from text by matching
    against JOB_TITLES list.
    Returns a list of matched titles.
    """
    found_titles = []
    text_lower   = text.lower()

    for title in JOB_TITLES:
        pattern = r'\b' + re.escape(title.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_titles.append(title)

    return found_titles


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
    Returns a score from 0 to 100.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    if not jd_skills:
        return 0

    resume_skill_set = set(s.lower() for s in resume_skills)
    matched_count    = sum(1 for s in jd_skills if s.lower() in resume_skill_set)

    return round((matched_count / len(jd_skills)) * 100, 2)


def calculate_title_score(resume_text, jd_text):
    """
    Score 4 — Job Title Match Score.
    Lightweight alternative to full NER.
    Checks if the job titles in the JD
    are present or closely related in the resume.

    Logic:
    - If resume contains exact JD title → 100%
    - If resume contains a related title → 60%
      (e.g. "ML Engineer" vs "AI Engineer")
    - If no title found in JD → neutral 50%
    - If no match at all → 0%

    Returns a score from 0 to 100.
    """
    resume_titles = extract_job_titles(resume_text)
    jd_titles     = extract_job_titles(jd_text)

    # If JD has no detectable title give neutral score
    if not jd_titles:
        return 50

    resume_title_set = set(t.lower() for t in resume_titles)

    exact_matches   = 0
    related_matches = 0

    for jd_title in jd_titles:
        jd_lower = jd_title.lower()

        # Exact match
        if jd_lower in resume_title_set:
            exact_matches += 1
            continue

        # Related match — check if key words overlap
        # e.g. "ML Engineer" and "Machine Learning Engineer"
        jd_words = set(jd_lower.split())
        for resume_title in resume_titles:
            resume_words = set(resume_title.lower().split())
            common_words = jd_words & resume_words
            # If more than half the words match → related
            if len(common_words) >= max(1, len(jd_words) // 2):
                related_matches += 1
                break

    total = len(jd_titles)
    score = ((exact_matches * 1.0) + (related_matches * 0.6)) / total * 100
    return round(min(score, 100), 2)


def calculate_hybrid_score(resume_text, jd_text):
    """
    Final Score — Four-way Hybrid.

    Combines four scoring methods:
      - Skill Overlap  (50%) : % of JD skills found in resume
      - Semantic       (20%) : conceptual similarity
      - Title Match    (20%) : job title alignment
      - TF-IDF         (10%) : exact keyword matching

    Skill overlap gets highest weight as it's
    the most direct and meaningful ATS metric.
    TF-IDF gets lowest weight as it naturally
    scores low on long documents.

    Formula:
        (0.5 × skill) + (0.2 × semantic) +
        (0.2 × title) + (0.1 × tfidf)

    Returns:
        final_score   : weighted hybrid score
        tfidf_score   : raw TF-IDF score
        semantic_score: raw semantic score
        skill_score   : raw skill overlap score
        title_score   : raw title match score
    """
    tfidf_score    = calculate_tfidf_score(resume_text, jd_text)
    semantic_score = calculate_semantic_score(resume_text, jd_text)
    skill_score    = calculate_skill_score(resume_text, jd_text)
    title_score    = calculate_title_score(resume_text, jd_text)

    final_score = round(
        (0.5 * skill_score)   +
        (0.2 * semantic_score) +
        (0.2 * title_score)   +
        (0.1 * tfidf_score),
        2
    )

    return final_score, tfidf_score, semantic_score, skill_score, title_score


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