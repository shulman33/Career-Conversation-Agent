import sqlite3
from typing import List, Dict

DB_PATH = "me/qa_database.db"


def init_database():
    """Initialize the Q&A database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def fetch_all_qa() -> List[Dict[str, str]]:
    """Fetch all Q&A pairs from the database."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM qa ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"question": q, "answer": a} for q, a in rows]


def insert_qa(question: str, answer: str):
    """Insert a new Q&A pair into the database."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO qa (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()


def update_qa(question: str, new_answer: str) -> bool:
    """Update the answer for an existing question. Returns True if updated."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE qa
        SET answer = ?
        WHERE question = ?
        AND id = (
            SELECT id FROM qa WHERE question = ? ORDER BY created_at DESC LIMIT 1
        )
    """, (new_answer, question, question))

    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def parse_qa_from_summary(file_path: str = "me/summary.md") -> List[Dict[str, str]]:
    """Parse Q&A pairs from summary.md markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    qa_pairs = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for ### headers (questions)
        if line.startswith("### ") and not line.startswith("####"):
            question = line[4:].strip()

            # Collect answer lines until next header or end
            answer_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith("#"):
                    break
                if next_line or answer_lines:
                    answer_lines.append(lines[i].rstrip())
                i += 1

            answer = "\n".join(answer_lines).strip()

            if answer:
                qa_pairs.append({"question": question, "answer": answer})

            continue

        i += 1

    return qa_pairs


def get_resume_linkedin_qa() -> List[Dict[str, str]]:
    """Return additional Q&A pairs extracted from resume and LinkedIn data."""
    return [
        {
            "question": "What certifications do you have?",
            "answer": """I hold several certifications:
- Programming with JavaScript
- React Basics
- Building Data Lakes on AWS
- AWS Cloud Technical Essentials
- Architecting Solutions on AWS
- Currently pursuing HL7 FHIR Certification"""
        },
        {
            "question": "What languages do you speak?",
            "answer": "I'm fluent in English and have professional working proficiency in Hebrew."
        },
        {
            "question": "What was your research assistant role about?",
            "answer": """At Katz School at Yeshiva University (August 2022 - January 2023), I worked as a Research Assistant in the New York City Metropolitan Area. I was tasked with deploying microservice benchmarks and carrying out DDoS attacks for security research purposes. This gave me hands-on experience with distributed systems, security testing, and performance benchmarking."""
        },
        {
            "question": "What did you work on at SIDEARM Sports?",
            "answer": """At SIDEARM Sports (April 2023 - October 2023), I worked as a Jr. Developer in Syracuse, NY. Key accomplishments:
- Collaborated in an Agile environment with daily stand-ups, sprint planning, and retrospectives
- Played a significant role in overhauling the company's CMS product from monolithic to microservice-based architecture
- Contributed to a database redesign that consolidated 1,300 client-specific databases into a streamlined 12-database system
- Enhanced user experience on NCAA Tickets website by fixing mobile device layout issues
- Developed a search and filter feature for the UConn Huskies streaming and on-demand service"""
        },
        {
            "question": "What is your contact information?",
            "answer": """You can reach me at:
- Email: samshulman6@gmail.com
- LinkedIn: linkedin.com/in/sam-shulman
- GitHub: github.com/shulman33
- Portfolio: www.samjshulman.com"""
        },
        {
            "question": "What awards or honors have you received?",
            "answer": """I've received the following honors and awards:
- Dean's List at Yeshiva University
- Second Place Hackathon Winner"""
        },
        {
            "question": "What is your experience with distributed systems?",
            "answer": """I have significant experience with distributed systems:
- Bachelor's degree with a Distributed Systems Track at Yeshiva University
- Relevant coursework: Distributed Systems, Parallel Programming, Algorithms, Operating Systems, Networking, Compilers
- Research Assistant experience deploying microservice benchmarks
- Professional experience transitioning monolithic systems to microservices at SIDEARM Sports
- Current work at Healthfirst designing event-driven serverless architectures"""
        },
        {
            "question": "Tell me about the ImIn project",
            "answer": """ImIn was an Automated Course Registration System I built from January to June 2023:
- Developed a registration system that automated class enrollment in less than one second when slots opened
- Achieved 2,256 page views and 946 unique page views in 30 days
- Served roughly 50% of the student body at Yeshiva University
- Solved a real pain point for students dealing with competitive course registration"""
        },
        {
            "question": "What big data experience do you have?",
            "answer": """I have hands-on experience with big data at multiple scales:
- At Cognizant: Worked with petabytes of healthcare data to develop Power BI reports
- At Healthfirst: Supporting the team in processing Machine-Readable Files (MRFs) ranging from 100 GB to 1 TB for federal compliance initiatives
- Skills in SQL optimization that improved report visualization loading time by 80%"""
        },
        {
            "question": "What frontend technologies do you know?",
            "answer": """My frontend technology stack includes:
- Languages: TypeScript, JavaScript, HTML, CSS
- Frameworks: React, Next.js, Vue.js
- UI Libraries: Antd, Shadcn UI, Tailwind CSS
- Tools: Git, GitHub, Jira
I've built administrative dashboards, user-facing applications, and Chrome extensions using these technologies."""
        },
        {
            "question": "What backend technologies do you know?",
            "answer": """My backend technology stack includes:
- Languages: Python, Java, SQL
- Databases: PostgreSQL, SQLite
- Frameworks: Django, FastAPI, Wagtail, Node.js
- Cloud: AWS (Step Functions, SES, Lambda, S3)
- Tools: Docker, JUnit
I've built APIs, data pipelines, and serverless architectures using these technologies."""
        },
    ]


def seed_database():
    """Seed the database with Q&A pairs from summary.md and resume/LinkedIn if empty."""
    qa_pairs = fetch_all_qa()
    if len(qa_pairs) == 0:
        # First, seed from summary.md
        summary_qa = parse_qa_from_summary()
        print(f"Seeding database with {len(summary_qa)} Q&A pairs from summary.md")
        for qa in summary_qa:
            insert_qa(qa["question"], qa["answer"])

        # Then, add resume/LinkedIn specific Q&A pairs
        resume_qa = get_resume_linkedin_qa()
        print(f"Adding {len(resume_qa)} Q&A pairs from resume/LinkedIn data")
        for qa in resume_qa:
            insert_qa(qa["question"], qa["answer"])
