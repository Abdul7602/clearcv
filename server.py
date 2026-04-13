import os
import io
import json
import re
from datetime import date
import httpx
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─── System Prompts ────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are ClearCV, an expert AI resume reviewer. You analyze resumes against job descriptions and provide structured, actionable feedback.

When given a resume and job description, return your analysis in EXACTLY this JSON format (no markdown, no code fences, just raw JSON):

{
  "match_score": <number 0-100>,
  "match_summary": "<one sentence explaining the score>",
  "ats_check": {
    "score": "<Good|Fair|Poor>",
    "issues": ["<issue 1>", "<issue 2>"]
  },
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "rewrite_suggestions": [
    {
      "original": "<exact line from resume>",
      "suggested": "<improved version>",
      "reason": "<why this is better>"
    }
  ],
  "missing_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>", "<keyword 5>"]
}

Rules:
- Be specific and actionable, not generic
- Reference actual content from the resume and job description
- Match score should reflect real alignment, not be generous
- Missing keywords should come directly from the job description
- Rewrite suggestions should reference actual lines from the resume
- ATS issues should flag real formatting/keyword problems
- Keep each strength/weakness to 1-2 sentences
- Return ONLY valid JSON, no other text"""

COVER_LETTER_PROMPT = """You are ClearCV, an expert cover letter writer. You write tailored, professional cover letters based on a candidate's resume and a specific job description.

Analyse the job description carefully:
- Identify what the employer emphasises most (technical skills, leadership, culture fit, specific tools, domain expertise, etc.)
- The cover letter MUST mirror those priorities — if they emphasise technical depth, lead with technical achievements; if they emphasise teamwork, lead with collaboration examples; if they list specific tools/technologies, reference them explicitly.

Return ONLY this JSON format (no markdown, no code fences):

{
  "candidate_name": "<full name extracted from resume, or 'Your Name' if not found>",
  "company_name": "<company name from job description, or 'the Company' if not found>",
  "role": "<exact job title from job description>",
  "paragraphs": [
    "<Opening paragraph: strong hook, name the role and company, show genuine enthusiasm based on something specific in the job description>",
    "<Body paragraph 1: 2-3 specific achievements from the resume that directly address the top requirements in the job description. Use numbers/metrics where available.>",
    "<Body paragraph 2: address any other key requirements from the job description, cultural fit, or why this specific company/role excites the candidate>",
    "<Closing paragraph: confident call to action, express eagerness to discuss further>"
  ]
}

Rules:
- Write in first person, professional but warm tone
- Each paragraph should be 3-5 sentences
- Never use generic filler phrases like 'I am a hard worker' or 'I am passionate about'
- Ground every claim in something from the resume or job description
- If the job emphasises specific tools, mention them by name
- Return ONLY valid JSON, no other text"""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/api/parse-cv")
async def parse_cv(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
        elif filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            return JSONResponse({"error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file."}, status_code=400)

        text = text.strip()
        if not text:
            return JSONResponse({"error": "Could not extract text from this file. Try pasting the text manually."}, status_code=400)
        return {"text": text}
    except Exception as e:
        return JSONResponse({"error": f"Failed to parse file: {str(e)}"}, status_code=500)


@app.post("/api/extract-url")
async def extract_url(request: Request):
    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "URL is required."}, status_code=400)
    if not url.startswith("http"):
        url = "https://" + url
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
            response = await http.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        selectors = [
            {"id": "job-description"},
            {"class": re.compile(r"job.?description|jobDescription|job.?detail|jobDetail", re.I)},
            {"class": re.compile(r"description__text|job-view-layout|show-more-less-html", re.I)},
        ]
        container = None
        for sel in selectors:
            container = soup.find(attrs=sel)
            if container:
                break
        if not container:
            container = soup.find("main") or soup.find("article") or soup.body

        if container:
            lines = [l.strip() for l in container.get_text(separator="\n").split("\n") if l.strip()]
            text = "\n".join(lines)
        else:
            text = ""

        if not text or len(text) < 100:
            return JSONResponse({"error": "Could not extract job description from this URL. Try pasting the text manually."}, status_code=400)

        return {"text": text[:8000]}
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": f"Could not access that URL (HTTP {e.response.status_code}). Try pasting the job description manually."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to fetch URL: {str(e)}"}, status_code=500)


@app.post("/api/review")
async def review_resume(request: Request):
    body = await request.json()
    resume = body.get("resume", "")
    job_description = body.get("job_description", "")
    if not resume or not job_description:
        return {"error": "Both resume and job description are required."}

    user_message = f"Please review this resume against the job description.\n\n---RESUME---\n{resume}\n\n---JOB DESCRIPTION---\n{job_description}"

    async def generate():
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": user_message}],
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/cover-letter")
async def generate_cover_letter(request: Request):
    body = await request.json()
    resume = body.get("resume", "")
    job_description = body.get("job_description", "")
    if not resume or not job_description:
        return JSONResponse({"error": "Resume and job description are required."}, status_code=400)

    user_message = f"Write a tailored cover letter for this candidate.\n\n---RESUME---\n{resume}\n\n---JOB DESCRIPTION---\n{job_description}"

    async def generate():
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": COVER_LETTER_PROMPT}, {"role": "user", "content": user_message}],
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/download/pdf")
async def download_pdf(request: Request):
    body = await request.json()
    cl = body.get("cover_letter", {})

    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=1.25 * inch, leftMargin=1.25 * inch,
        topMargin=1 * inch, bottomMargin=1 * inch
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Normal"], fontSize=15, fontName="Helvetica-Bold", spaceAfter=2)
    date_style  = ParagraphStyle("Date",  parent=styles["Normal"], fontSize=11, spaceAfter=18)
    open_style  = ParagraphStyle("Open",  parent=styles["Normal"], fontSize=11, spaceAfter=16)
    body_style  = ParagraphStyle("Body",  parent=styles["Normal"], fontSize=11, leading=17, spaceAfter=14, alignment=TA_LEFT)
    close_style = ParagraphStyle("Close", parent=styles["Normal"], fontSize=11, spaceBefore=16)

    story = [
        Paragraph(cl.get("candidate_name", ""), name_style),
        Paragraph(date.today().strftime("%B %d, %Y"), date_style),
        Paragraph(f"Re: {cl.get('role', 'Application')} — {cl.get('company_name', '')}", date_style),
        Spacer(1, 6),
        Paragraph("Dear Hiring Manager,", open_style),
    ]

    for para in cl.get("paragraphs", []):
        story.append(Paragraph(para, body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Sincerely,", close_style))
    story.append(Spacer(1, 36))
    story.append(Paragraph(cl.get("candidate_name", ""), close_style))

    doc.build(story)
    buffer.seek(0)

    filename = f"Cover_Letter_{cl.get('company_name', 'Application').replace(' ', '_')}.pdf"
    return Response(
        content=buffer.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/api/download/docx")
async def download_docx(request: Request):
    body = await request.json()
    cl = body.get("cover_letter", {})

    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Margins
    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    # Candidate name
    name_para = doc.add_paragraph()
    name_run = name_para.add_run(cl.get("candidate_name", ""))
    name_run.bold = True
    name_run.font.size = Pt(15)
    name_para.paragraph_format.space_after = Pt(2)

    # Date + Re line
    date_para = doc.add_paragraph(date.today().strftime("%B %d, %Y"))
    date_para.paragraph_format.space_after = Pt(2)
    date_para.runs[0].font.size = Pt(11)

    re_para = doc.add_paragraph(f"Re: {cl.get('role', 'Application')} — {cl.get('company_name', '')}")
    re_para.paragraph_format.space_after = Pt(14)
    re_para.runs[0].font.size = Pt(11)

    # Salutation
    sal = doc.add_paragraph("Dear Hiring Manager,")
    sal.paragraph_format.space_after = Pt(12)
    sal.runs[0].font.size = Pt(11)

    # Body paragraphs
    for para_text in cl.get("paragraphs", []):
        p = doc.add_paragraph(para_text)
        p.paragraph_format.space_after = Pt(12)
        for run in p.runs:
            run.font.size = Pt(11)

    # Closing
    doc.add_paragraph()
    closing = doc.add_paragraph("Sincerely,")
    closing.runs[0].font.size = Pt(11)

    doc.add_paragraph()
    doc.add_paragraph()

    sig = doc.add_paragraph(cl.get("candidate_name", ""))
    sig.runs[0].font.size = Pt(11)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    filename = f"Cover_Letter_{cl.get('company_name', 'Application').replace(' ', '_')}.docx"
    return Response(
        content=buffer.read(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
