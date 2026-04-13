import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are ClearCV, an expert AI resume reviewer. You analyze resumes against job descriptions and provide structured, actionable feedback.

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


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/api/review")
async def review_resume(request: Request):
    body = await request.json()
    resume = body.get("resume", "")
    job_description = body.get("job_description", "")

    if not resume or not job_description:
        return {"error": "Both resume and job description are required."}

    user_message = f"""Please review this resume against the job description.

---RESUME---
{resume}

---JOB DESCRIPTION---
{job_description}"""

    async def generate():
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
