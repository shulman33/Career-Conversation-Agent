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


def seed_database():
    """Seed the database with Q&A pairs from summary.md if it's empty."""
    qa_pairs = fetch_all_qa()
    if len(qa_pairs) == 0:
        summary_qa = parse_qa_from_summary()
        print(f"Seeding database with {len(summary_qa)} Q&A pairs from summary.md")
        for qa in summary_qa:
            insert_qa(qa["question"], qa["answer"])
