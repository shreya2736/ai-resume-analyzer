def get_feedback_prompt(resume_text, job_description, ats_score, missing_skills):
    """
    Simple, focused feedback prompt.
    """
    missing_skills_str = ", ".join(missing_skills) if missing_skills else "None"

    prompt = f"""
You are an expert resume coach. Be direct and specific. No filler words.

ATS Score: {ats_score}%
Missing Skills: {missing_skills_str}

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}

Give feedback in exactly this format:

**OVERALL ASSESSMENT**
2-3 sentences only. How well does this resume match the job?

**WHAT TO IMPROVE IN YOUR RESUME**
List exactly 4 specific things to fix in the resume.
For each point mention which section (Summary, Skills, Experience, Projects).
Format: - [Section]: What to fix and why.

**SKILLS TO LEARN**
List the top 5 missing skills ranked by importance for this job.
For each skill add one line on how to learn it quickly.
Format: - [Skill]: How to learn it.

**TOP 3 ACTION ITEMS**
The 3 most impactful changes to make right now.
Format: 1. Action item

Keep everything short, specific, and actionable.
No generic advice. Reference actual content from the resume.
"""
    return prompt


def get_rewriter_prompt(project_description):
    """
    Prompt for rewriting weak project descriptions.
    """
    prompt = f"""
You are an expert resume writer for tech resumes.

Rewrite the following into a strong resume bullet point.

Rules:
- Start with a strong action verb
- Include technologies mentioned
- Quantify impact if possible
- Maximum 2 lines
- ATS-friendly

Original:
{project_description}

Return only the rewritten bullet point. Nothing else.
"""
    return prompt


def get_role_recommendation_prompt(resume_text):
    """
    Prompt for recommending suitable job roles.
    """
    prompt = f"""
You are a career advisor for tech professionals.

Analyze this resume and recommend suitable job roles.

--- RESUME ---
{resume_text}

Give recommendations in exactly this format:

**TOP 5 RECOMMENDED ROLES**
For each role provide:
- Role name
- Match percentage (estimate based on resume skills)
- Why this role fits (1 line)
- One skill to add to be more competitive

Format each role like this:
**[Role Name]** — [Match]% match
Why: [one line reason]
Upskill: [one skill to add]

**BEST FITTING DOMAIN**
Which domain suits this person most and why (2 lines max).

**CAREER PATH SUGGESTION**
Short term (0-1 year) and long term (2-3 years) goals. 2 lines each.

Be specific to the actual skills in the resume.
"""
    return prompt


def get_report_prompt(resume_text, job_description, ats_score, missing_skills, common_skills, feedback):
    """
    Prompt for generating a clean downloadable report summary.
    """
    missing_str = ", ".join(missing_skills) if missing_skills else "None"
    common_str = ", ".join(common_skills) if common_skills else "None"

    prompt = f"""
Create a clean resume analysis report summary.

Data:
- ATS Score: {ats_score}%
- Matched Skills: {common_str}
- Missing Skills: {missing_str}

Feedback already given:
{feedback}

Write a professional report in this format:

RESUME ANALYSIS REPORT
======================

ATS MATCH SCORE: {ats_score}%

MATCHED SKILLS:
{common_str}

MISSING SKILLS:
{missing_str}

KEY FINDINGS:
[3 bullet points summarizing the analysis]

IMPROVEMENT AREAS:
[4 specific areas from the resume to improve]

SKILLS TO LEARN:
[Top 5 skills with one line each]

ACTION PLAN:
[3 clear next steps]

Keep it clean and professional. Plain text only, no markdown symbols.
"""
    return prompt