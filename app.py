import streamlit as st
from utils.parser import extract_text_from_pdf
from utils.matcher import (
    extract_skills,
    calculate_hybrid_score,
    get_missing_skills,
    get_common_skills
)
from utils.groq_api import (
    configure_groq,
    get_resume_feedback,
    rewrite_project,
    get_role_recommendations,
    generate_report
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

.main { padding-top: 1rem; }
            
/* Global dark theme override */
html, body, [data-testid="stAppViewContainer"], .main, .stApp {
    background-color: #0e1117 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #0e1117 !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1c29 !important;
}
body, .stMarkdown, .stText, .stTextInput, .stTextArea, label {
    color: #e2e8f0 !important;
}
h1, h2, h3, h4, h5, h6, .section-header {
    color: #e94560 !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #1e1e2e !important;
    color: #e2e8f0 !important;
    border-color: #3d3d5c !important;
}
.stButton button {
    background-color: #e94560 !important;
    color: white !important;
}
.streamlit-expanderHeader {
    background-color: #1e1e2e !important;
    color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #1e1e2e !important;
}

.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #0f3460;
}
.header-banner h1 {
    color: #e94560;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}
.header-banner p {
    color: #a8b2d8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.metric-container {
    background: linear-gradient(135deg, #1e1e2e, #2d2d44);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #3d3d5c;
    margin-bottom: 1rem;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #e94560;
    margin: 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #a8b2d8;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.score-bar-container {
    background: #2d2d44;
    border-radius: 50px;
    height: 14px;
    width: 100%;
    margin: 1rem 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 14px;
    border-radius: 50px;
}

.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.5rem;
}
.badge-green {
    background: #1a3a2a;
    color: #4ade80;
    border: 1px solid #4ade80;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-red {
    background: #3a1a1a;
    color: #f87171;
    border: 1px solid #f87171;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e94560;
    display: inline-block;
}

.status-success {
    background: #1a3a2a;
    border-left: 4px solid #4ade80;
    color: #4ade80;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-weight: 600;
}
.status-warning {
    background: #3a2e1a;
    border-left: 4px solid #fbbf24;
    color: #fbbf24;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-weight: 600;
}
.status-error {
    background: #3a1a1a;
    border-left: 4px solid #f87171;
    color: #f87171;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-weight: 600;
}

.rewriter-original {
    background: #1e1e2e;
    border-left: 4px solid #a8b2d8;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    color: #a8b2d8;
    margin: 0.5rem 0;
}
.rewriter-result {
    background: #1a3a2a;
    border-left: 4px solid #4ade80;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    color: #4ade80;
    margin: 0.5rem 0;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #4a4a6a;
    font-size: 0.8rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #2d2d44;
    margin-top: 2rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1e1e2e;
    padding: 0.5rem;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    color: #a8b2d8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #e94560 !important;
    color: white !important;
}

/* Expander content - make text visible */
.streamlit-expanderContent {
    background-color: #1e1e2e !important;
    color: #e2e8f0 !important;
}

/* The actual text inside the expander (from st.text) */
.streamlit-expanderContent pre,
.streamlit-expanderContent code,
.streamlit-expanderContent .stText {
    color: #e2e8f0 !important;
    background-color: #1e1e2e !important;
    border: none !important;
}

/* Any st.text output anywhere else */
.stText {
    color: #e2e8f0 !important;
}

/* For the text area placeholder */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #b0b0d0 !important;
    opacity: 1 !important;
}
            
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def render_score_bar(score):
    color = "#4ade80" if score >= 70 else "#fbbf24" if score >= 40 else "#f87171"
    st.markdown(f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width:{score}%; background:{color};"></div>
    </div>
    """, unsafe_allow_html=True)


def render_badges(skills, badge_type="green"):
    if not skills:
        st.markdown("<p style='color:#4a4a6a;'>None found</p>", unsafe_allow_html=True)
        return
    badges = " ".join([f'<span class="badge-{badge_type}">{s}</span>' for s in skills])
    st.markdown(f'<div class="badge-container">{badges}</div>', unsafe_allow_html=True)


def render_metric(value, label):
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def check_api_key():
    """Returns api_key from session state or empty string."""
    return st.session_state.get('api_key', '')


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📄 AI Resume Analyzer</h1>
    <p>Upload your resume · Paste a job description · Get ATS score, AI feedback & role recommendations</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Try user's own key first
    user_key = st.text_input(
        "🔑 Your Groq API Key (optional)",
        type="password",
        placeholder="Paste your own key for unlimited use",
        key="api_key_input"
    )

    # Fall back to shared key from secrets
    try:
        fallback_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        fallback_key = ""

    if user_key:
        api_key = user_key
        st.session_state['api_key'] = api_key
        st.success("✅ Using your API key!")
    elif fallback_key:
        api_key = fallback_key
        st.session_state['api_key'] = api_key
        st.info("ℹ️ Using shared key — AI features ready!")
    else:
        api_key = ""
        st.session_state['api_key'] = ""
        st.warning("⚠️ No API key available.")

    st.markdown("---")
    st.markdown("**📌 Want unlimited use?**")
    st.markdown("Get your own free key:")
    st.markdown("1. Visit [console.groq.com](https://console.groq.com)")
    st.markdown("2. Sign up — no credit card needed")
    st.markdown("3. API Keys → Create Key → Paste above")

    st.markdown("---")
    st.markdown("**🤖 Model:** Llama 3.3 70B via Groq")
        

    st.markdown("---")
    st.markdown("**✅ Features**")
    st.markdown("- ATS Score & Skill Matching")
    st.markdown("- Resume Improvement Suggestions")
    st.markdown("- Skills to Learn")
    st.markdown("- Role Recommendations")
    st.markdown("- Resume Rewriter")
    st.markdown("- Downloadable Report")


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">📤 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your PDF resume",
        type=["pdf"],
        help="Upload a PDF version of your resume"
    )

    if uploaded_file is not None:
        with st.spinner("Extracting resume text..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:
            st.session_state['resume_text'] = resume_text
            st.markdown('<div class="status-success">✔ Resume parsed successfully</div>',
                       unsafe_allow_html=True)
            with st.expander("👁 Preview Extracted Text"):
                st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
        else:
            st.markdown('<div class="status-error">✘ Could not extract text. Try another PDF.</div>',
                       unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the job description here",
        height=280,
        placeholder="e.g. Looking for a Python developer with SQL, Machine Learning experience...",
        help="Copy and paste the full job description"
    )
    if job_description:
        st.session_state['job_description'] = job_description
        word_count = len(job_description.split())
        st.markdown(f'<div class="status-success">✔ Job description ready · {word_count} words</div>',
                   unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
both_ready = uploaded_file is not None and job_description.strip() != ""
analyze_btn = st.button(
    "🔍 Analyze My Resume",
    disabled=not both_ready,
    use_container_width=True,
    type="primary"
)


# ─────────────────────────────────────────────
# ANALYSIS LOGIC
# ─────────────────────────────────────────────
if analyze_btn:
    with st.spinner("Analyzing your resume..."):
        resume_skills  = extract_skills(resume_text)
        jd_skills      = extract_skills(job_description)
        ats_score, tfidf_score, semantic_score, skill_score = calculate_hybrid_score(resume_text, job_description)
        common_skills  = get_common_skills(resume_skills, jd_skills)
        missing_skills = get_missing_skills(resume_skills, jd_skills)

        st.session_state['tfidf_score']    = tfidf_score
        st.session_state['semantic_score'] = semantic_score
        st.session_state['skill_score']    = skill_score
        st.session_state['ats_score']      = ats_score
        st.session_state['missing_skills'] = missing_skills
        st.session_state['resume_skills']  = resume_skills
        st.session_state['jd_skills']      = jd_skills
        st.session_state['common_skills']  = common_skills
        st.session_state['analysis_done']  = True
        # Clear old feedback when re-analyzing
        st.session_state.pop('feedback', None)
        st.session_state.pop('report', None)
        st.session_state.pop('roles', None)


# ─────────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────────

if st.session_state.get('analysis_done'):

    ats_score       = st.session_state['ats_score']
    missing_skills  = st.session_state['missing_skills']
    resume_skills   = st.session_state['resume_skills']
    jd_skills       = st.session_state['jd_skills']
    common_skills   = st.session_state['common_skills']
    resume_text     = st.session_state.get('resume_text', '')
    job_description = st.session_state.get('job_description', '')
    api_key         = check_api_key()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  ATS Score",
    "💡  AI Feedback",
    "🎓  Role Recommendations",
    "📋  All Skills",
    "📊  Score Breakdown",
    ])

    # ── TAB 1: ATS Score ─────────────────────────────────────
    with tab1:

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric(f"{ats_score}%", "ATS Match Score")
        with m2:
            render_metric(len(common_skills), "Skills Matched")
        with m3:
            render_metric(len(missing_skills), "Skills Missing")

        render_score_bar(ats_score)

        if ats_score >= 70:
            st.markdown('<div class="status-success">🟢 Strong match — your resume aligns well with this job!</div>',
                       unsafe_allow_html=True)
        elif ats_score >= 40:
            st.markdown('<div class="status-warning">🟡 Moderate match — some improvements recommended.</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">🔴 Low match — tailor your resume for this role.</div>',
                       unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2 = st.columns(2, gap="large")

        with sc1:
            st.markdown('<div class="section-header">✅ Matched Skills</div>', unsafe_allow_html=True)
            render_badges(common_skills, "green")

        with sc2:
            st.markdown('<div class="section-header">❌ Missing Skills</div>', unsafe_allow_html=True)
            render_badges(missing_skills, "red")

    # ── TAB 2: AI Feedback ───────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">💡 Resume Improvement Feedback</div>',
                   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not api_key:
            st.markdown('<div class="status-warning">⚠️ Enter your Groq API key in the sidebar.</div>',
                       unsafe_allow_html=True)
        else:
            # Only call API once — cache in session state
            if 'feedback' not in st.session_state:
                with st.spinner("Getting AI feedback..."):
                    try:
                        model    = configure_groq(api_key)
                        feedback = get_resume_feedback(
                            model, resume_text, job_description,
                            ats_score, missing_skills
                        )
                        st.session_state['feedback'] = feedback
                    except Exception as e:
                        st.markdown(f'<div class="status-error">❌ {str(e)}</div>',
                                   unsafe_allow_html=True)

            if 'feedback' in st.session_state:
                st.markdown(st.session_state['feedback'])

    # ── TAB 3: Role Recommendations ──────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🎓 Recommended Job Roles</div>',
                   unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not api_key:
            st.markdown('<div class="status-warning">⚠️ Enter your Groq API key in the sidebar.</div>',
                       unsafe_allow_html=True)
        else:
            if 'roles' not in st.session_state:
                with st.spinner("Finding best roles for your profile..."):
                    try:
                        model = configure_groq(api_key)
                        roles = get_role_recommendations(model, resume_text)
                        st.session_state['roles'] = roles
                    except Exception as e:
                        st.markdown(f'<div class="status-error">❌ {str(e)}</div>',
                                   unsafe_allow_html=True)

            if 'roles' in st.session_state:
                st.markdown(st.session_state['roles'])

    # ── TAB 4: All Skills ────────────────────────────────────
    with tab4:
        ac1, ac2 = st.columns(2, gap="large")

        with ac1:
            st.markdown('<div class="section-header">📄 Skills in Your Resume</div>',
                       unsafe_allow_html=True)
            render_badges(resume_skills, "green")

        with ac2:
            st.markdown('<div class="section-header">📋 Skills in Job Description</div>',
                       unsafe_allow_html=True)
            render_badges(jd_skills, "green")
    
    # ── TAB 5: Score Breakdown ────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">📊 Score Breakdown</div>',
                   unsafe_allow_html=True)
        st.markdown("How your ATS score was calculated using the hybrid scoring method.")
        st.markdown("<br>", unsafe_allow_html=True)

        tfidf    = st.session_state.get('tfidf_score', 0)
        semantic = st.session_state.get('semantic_score', 0)
        skill    = st.session_state.get('skill_score', 0)
        final    = st.session_state.get('ats_score', 0)

        # Four metric cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_metric(f"{skill}%", "Skill Overlap")
        with s2:
            render_metric(f"{tfidf}%", "TF-IDF Score")
        with s3:
            render_metric(f"{semantic}%", "Semantic Score")
        with s4:
            render_metric(f"{final}%", "Final Score")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**🎯 Skill Overlap Score — JD Skills Found in Resume (60% weight)**")
        render_score_bar(skill)
        st.markdown("Percentage of required JD skills detected in your resume. Most direct ATS metric.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**🔤 TF-IDF Score — Keyword Matching (10% weight)**")
        render_score_bar(tfidf)
        st.markdown("Checks for exact keyword overlap between your resume and the JD.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**🧠 Semantic Score — Meaning Matching (30% weight)**")
        render_score_bar(semantic)
        st.markdown("Understands conceptual similarity even when exact words differ.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**⭐ Final Hybrid Score — Weighted Combination**")
        render_score_bar(final)
        st.markdown("Formula: **(0.6 × Skill) + (0.1 × TF-IDF) + (0.3 × Semantic)**")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("""
        **💡 How to read these scores:**
        - **Skill Overlap** is the most important — add missing JD skills to your resume to raise this.
        - **TF-IDF Score** improves when you mirror exact keywords from the JD.
        - **Semantic Score** reflects overall conceptual alignment with the job.
        - **Final Score** combines all three. Aim for above 50% for a strong match.
        """)


# ─────────────────────────────────────────────
# DOWNLOAD REPORT SECTION
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-header">📥 Download Analysis Report</div>', unsafe_allow_html=True)
st.markdown("Generate and download a complete analysis report of your resume.")
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.get('analysis_done'):
    st.markdown('<div class="status-warning">⚠️ Please analyze your resume first.</div>',
               unsafe_allow_html=True)
else:
    api_key = check_api_key()
    if not api_key:
        st.markdown('<div class="status-error">❌ Enter your Groq API key in the sidebar.</div>',
                   unsafe_allow_html=True)
    else:
        rp1, rp2 = st.columns([1, 1], gap="large")

        with rp1:
            st.markdown("**What's included in the report:**")
            st.markdown("- ✅ ATS Match Score")
            st.markdown("- ✅ Matched & Missing Skills")
            st.markdown("- ✅ Key Findings")
            st.markdown("- ✅ Resume Improvement Areas")
            st.markdown("- ✅ Skills to Learn")
            st.markdown("- ✅ Action Plan")

            generate_btn = st.button(
                "📄 Generate Report",
                use_container_width=True,
                type="primary"
            )

            if generate_btn:
                with st.spinner("Generating your report..."):
                    try:
                        model    = configure_groq(api_key)
                        feedback = st.session_state.get('feedback', 'Not generated yet.')
                        report   = generate_report(
                            model,
                            st.session_state.get('resume_text', ''),
                            st.session_state.get('job_description', ''),
                            st.session_state['ats_score'],
                            st.session_state['missing_skills'],
                            st.session_state['common_skills'],
                            feedback
                        )
                        st.session_state['report'] = report
                    except Exception as e:
                        st.markdown(f'<div class="status-error">❌ {str(e)}</div>',
                                   unsafe_allow_html=True)

        with rp2:
            if 'report' in st.session_state:
                report = st.session_state['report']

                with st.expander("👁 Preview Report"):
                    st.text(report)

                st.download_button(
                    label="⬇️ Download Report as .txt",
                    data=report,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                st.markdown('<div class="status-success">✔ Report ready to download!</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background:#1e1e2e;
                    border: 2px dashed #2d2d44;
                    border-radius: 12px;
                    padding: 2rem;
                    text-align: center;
                    color: #4a4a6a;
                    height: 180px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    📄 Your report will appear here
                </div>
                """, unsafe_allow_html=True)  

# ─────────────────────────────────────────────
# RESUME REWRITER SECTION
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-header">✍️ Resume Project Rewriter</div>', unsafe_allow_html=True)
st.markdown("Transform weak project descriptions into strong, ATS-friendly bullet points.")
st.markdown("<br>", unsafe_allow_html=True)

rw1, rw2 = st.columns([1, 1], gap="large")

with rw1:
    project_input = st.text_area(
        "📝 Your project description",
        height=180,
        placeholder="e.g. Built a weather app using Python",
        key="rewriter_input"
    )
    rewrite_btn = st.button(
        "✨ Rewrite Professionally",
        disabled=project_input.strip() == "",
        use_container_width=True,
        type="primary"
    )

with rw2:
    if rewrite_btn:
        api_key = check_api_key()
        if not api_key:
            st.markdown('<div class="status-error">❌ Enter your Groq API key in the sidebar.</div>',
                       unsafe_allow_html=True)
        else:
            with st.spinner("Rewriting..."):
                try:
                    model     = configure_groq(api_key)
                    rewritten = rewrite_project(model, project_input)

                    st.markdown("**Original:**")
                    st.markdown(f'<div class="rewriter-original">{project_input}</div>',
                               unsafe_allow_html=True)
                    st.markdown("**✨ Rewritten:**")
                    st.markdown(f'<div class="rewriter-result">{rewritten}</div>',
                               unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="status-error">❌ {str(e)}</div>',
                               unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background:#1e1e2e;
            border: 2px dashed #2d2d44;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            color: #4a4a6a;
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            ✨ Your rewritten bullet point will appear here
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Powered by Groq (Llama 3.3 70B) · AI Resume Analyzer
</div>
""", unsafe_allow_html=True)