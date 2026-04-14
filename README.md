<div align="center">

<!-- Replace with your crystal logo image once hosted -->
<img src="crystal_final.svg" alt="Crystal Career Logo" width="80"/>

# Crystal Career

### *Land your next role — AI resume analysis, ATS scoring, and tailored cover letters in seconds.*

**Live demo:** [clearcv.onrender.com](https://clearcv.onrender.com)

</div>

---

## Features

- **CV Upload** — drag & drop or click to upload PDF, DOCX, or TXT. Falls back to paste
- **Job Description Input** — paste text or fetch directly from a job posting URL (Indeed, Greenhouse, Lever, Workday)
- **Match Score** — animated score ring showing how well the CV fits the role (0–100)
- **ATS Compatibility Check** — flags missing keywords, formatting issues, and section header problems
- **Strengths & Weaknesses** — specific, grounded feedback referencing actual resume content
- **Rewrite Suggestions** — before/after line rewrites with reasoning
- **Missing Keywords** — top keywords from the job description not present in the CV
- **Tailored Cover Letter** — generated independently or alongside analysis, adjusts focus based on what the job description emphasises. Download as PDF or Word (.docx) in one click
- **Streaming responses** — output streams in real time, no waiting for the full result
- **Minimal dark UI** — clean single-page interface, no login, no database

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + FastAPI |
| AI | Groq API (LLaMA 3.3 70B) |
| Frontend | Vanilla JS + HTML/CSS (single file) |
| File Parsing | pypdf, python-docx |
| URL Extraction | httpx + BeautifulSoup4 |
| PDF Generation | ReportLab |
| Deployment | Render |

---

## Getting Started

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Abdul7602/clearcv.git
cd Crystal\ Career

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
# Windows
set GROQ_API_KEY=your-key-here

# Mac/Linux
export GROQ_API_KEY=your-key-here

# 4. Run the server
python server.py
```

Then open [http://localhost:8000](http://localhost:8000).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key — get one free at [console.groq.com](https://console.groq.com) |
| `PORT` | Port to run on (default: `8000`) — set automatically by Render |

---

## Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects settings from `render.yaml`:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py`
5. Add environment variable: `GROQ_API_KEY` → your key
6. Deploy

> **Note:** The free tier spins down after inactivity. First request after sleep may take ~50 seconds to wake up.

---

## Project Structure

```
crystal-career/
├── server.py        # FastAPI backend — all API endpoints
├── index.html       # Frontend — single file, vanilla JS
├── requirements.txt # Python dependencies
├── render.yaml      # Render deployment config
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/parse-cv` | Extracts text from uploaded PDF/DOCX/TXT |
| `POST` | `/api/extract-url` | Fetches and parses a job posting URL |
| `POST` | `/api/review` | Streams CV analysis as SSE |
| `POST` | `/api/cover-letter` | Streams cover letter generation as SSE |
| `POST` | `/api/download/pdf` | Returns cover letter as PDF file |
| `POST` | `/api/download/docx` | Returns cover letter as Word file |

---

## Built With

- [Groq](https://groq.com) — fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com) — Python web framework
- [Claude AI](https://claude.ai) — AI assistant used t