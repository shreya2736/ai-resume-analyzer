from groq import Groq
from utils.prompts import (
    get_feedback_prompt,
    get_rewriter_prompt,
    get_role_recommendation_prompt,
    get_report_prompt
)


def configure_groq(api_key):
    """
    Returns the API key.
    Groq client is created fresh in each function call.
    """
    return api_key


def _call_groq(api_key, system_message, user_prompt, max_tokens=1500):
    """
    Private helper — handles all Groq API calls.
    Avoids repeating the same boilerplate in every function.
    """
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content


def get_resume_feedback(model, resume_text, job_description, ats_score, missing_skills):
    """
    Get focused resume feedback with improvement areas
    and skills to learn.
    """
    prompt = get_feedback_prompt(resume_text, job_description, ats_score, missing_skills)
    try:
        return _call_groq(
            model,
            "You are an expert ATS resume coach. Be direct, specific and concise.",
            prompt,
            max_tokens=1200
        )
    except Exception as e:
        return f"Error getting feedback: {str(e)}"


def rewrite_project(model, project_description):
    """
    Rewrite a weak project description into a
    strong resume bullet point.
    """
    prompt = get_rewriter_prompt(project_description)
    try:
        return _call_groq(
            model,
            "You are an expert resume writer for tech resumes.",
            prompt,
            max_tokens=300
        )
    except Exception as e:
        return f"Error rewriting: {str(e)}"


def get_role_recommendations(model, resume_text):
    """
    Recommend suitable job roles based on
    the candidate's resume.
    """
    prompt = get_role_recommendation_prompt(resume_text)
    try:
        return _call_groq(
            model,
            "You are a career advisor for tech professionals.",
            prompt,
            max_tokens=1000
        )
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"


def generate_report(model, resume_text, job_description, ats_score, missing_skills, common_skills, feedback):
    """
    Generate a clean plain-text report
    suitable for download as a .txt file.
    """
    prompt = get_report_prompt(
        resume_text,
        job_description,
        ats_score,
        missing_skills,
        common_skills,
        feedback
    )
    try:
        return _call_groq(
            model,
            "You are a professional report writer. Write clean plain text reports.",
            prompt,
            max_tokens=1200
        )
    except Exception as e:
        return f"Error generating report: {str(e)}"