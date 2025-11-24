from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import sqlite3
from pydantic import BaseModel


# Pydantic model for evaluation responses
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


load_dotenv(override=True)

# Set up Gemini client for evaluation
gemini = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


# Database helper functions
DB_PATH = "me/qa_database.db"

def init_database():
    """Initialize the Q&A database if it doesn't exist"""
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

def fetch_all_qa():
    """Fetch all Q&A pairs from the database"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM qa ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"question": q, "answer": a} for q, a in rows]

def insert_qa(question, answer):
    """Insert a new Q&A pair into the database"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO qa (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def parse_qa_from_summary(file_path="me/summary.md"):
    """Parse Q&A pairs from summary.md markdown file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    qa_pairs = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for ### headers (questions)
        if line.startswith("### ") and not line.startswith("####"):
            question = line[4:].strip()  # Remove "### "

            # Collect answer lines until next header or end
            answer_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # Stop at next header of any level
                if next_line.startswith("#"):
                    break
                # Skip empty lines at start, but keep them in the middle
                if next_line or answer_lines:
                    answer_lines.append(lines[i].rstrip())
                i += 1

            # Join answer and clean up
            answer = "\n".join(answer_lines).strip()

            if answer:  # Only add if there's an answer
                qa_pairs.append({"question": question, "answer": answer})

            continue

        i += 1

    return qa_pairs

def seed_database():
    """Seed the database with Q&A pairs from summary.md if it's empty"""
    qa_pairs = fetch_all_qa()
    if len(qa_pairs) == 0:
        # Parse Q&A pairs from summary.md
        summary_qa = parse_qa_from_summary()
        print(f"Seeding database with {len(summary_qa)} Q&A pairs from summary.md")
        for qa in summary_qa:
            insert_qa(qa["question"], qa["answer"])


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    # Add question to database with placeholder answer
    placeholder_answer = "⚠️ ANSWER NEEDED - Please update this entry in the database"
    insert_qa(question, placeholder_answer)

    # Notify via push that an answer is needed
    push(f"❓ Question needs answer in database:\n\nQ: {question}\n\nPlease update the database with the correct answer.")
    return {"recorded": "ok", "added_to_database": True}

def search_qa_database(question):
    """Search the Q&A database for semantically similar questions"""
    qa_pairs = fetch_all_qa()

    if not qa_pairs:
        return {"found": False, "answer": None, "message": "Database is empty"}

    # Filter out questions with placeholder answers (questions awaiting answers)
    answered_qa_pairs = [qa for qa in qa_pairs if "⚠️ ANSWER NEEDED" not in qa['answer']]

    if not answered_qa_pairs:
        return {"found": False, "answer": None, "message": "No answered questions in database yet"}

    # Build context from answered Q&A pairs only
    context = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answered_qa_pairs])

    # Use OpenAI to find the best matching answer
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful assistant that matches user questions to a database of Q&A pairs.
Given a user's question and a database of Q&A pairs, determine if there's a semantically similar question in the database.
If there is a good match (the question is asking about the same topic, even if worded differently), respond with JSON in this format:
{{"found": true, "answer": "<the answer from the database>"}}

If there is no good match, respond with JSON in this format:
{{"found": false, "answer": null}}

Only match questions that are truly asking about the same information. Don't match if the topics are different.

Here is the Q&A database:
{context}"""
            },
            {
                "role": "user",
                "content": f"Does this question match any in the database? Question: {question}"
            }
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result

def add_qa_to_database(question, answer):
    """Add a new Q&A pair to the database"""
    insert_qa(question, answer)
    return {"added": True, "message": f"Successfully added Q&A pair to database"}

def list_recent_qa(limit=5):
    """List recent Q&A pairs from the database"""
    qa_pairs = fetch_all_qa()
    recent = qa_pairs[:limit]

    # Mark which ones need answers
    for qa in recent:
        qa['needs_answer'] = "⚠️ ANSWER NEEDED" in qa['answer']

    return {
        "count": len(recent),
        "qa_pairs": recent,
        "message": f"Retrieved {len(recent)} recent Q&A pairs"
    }

def update_qa_answer(question, new_answer):
    """Update the answer for a question in the database"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Update the most recent entry matching this question
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

    if rows_affected > 0:
        return {"updated": True, "message": f"Successfully updated answer for: {question}"}
    else:
        return {"updated": False, "message": f"Question not found: {question}"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

search_qa_database_json = {
    "name": "search_qa_database",
    "description": "Search the Q&A database for previously answered questions. Use this BEFORE answering any question to check if there's already a stored answer. This helps provide consistent, accurate responses.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question to search for in the database"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

add_qa_to_database_json = {
    "name": "add_qa_to_database",
    "description": "Add a new question and answer pair to the database. Use this to store commonly asked questions and their answers for future reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to store"
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question"
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}

list_recent_qa_json = {
    "name": "list_recent_qa",
    "description": "List recent Q&A pairs from the database. Useful for showing what questions have been answered before.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of Q&A pairs to retrieve (default: 5)"
            }
        },
        "required": [],
        "additionalProperties": False
    }
}

update_qa_answer_json = {
    "name": "update_qa_answer",
    "description": "Update the answer for an existing question in the database. Useful for replacing placeholder answers or correcting existing answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The exact question text to update"
            },
            "new_answer": {
                "type": "string",
                "description": "The new answer to store for this question"
            }
        },
        "required": ["question", "new_answer"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": search_qa_database_json},
    {"type": "function", "function": add_qa_to_database_json},
    {"type": "function", "function": list_recent_qa_json},
    {"type": "function", "function": update_qa_answer_json}
]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Sam Shulman"
        linkedin_reader = PdfReader("me/linkedin.pdf")
        resume_reader = PdfReader("me/resume.pdf")
        self.linkedin = ""
        self.resume = ""
        for page in resume_reader.pages:
            text = page.extract_text()
            if text:
                self.resume += text
        for page in linkedin_reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("me/summary.md", "r", encoding="utf-8") as f:
            self.summary = f.read()

        # Initialize and seed the Q&A database
        seed_database()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile and resume which you can use to answer questions. \
Be professional and engaging, as if talking to a recruiter or future employer who came across the website. \
\n\n\
IMPORTANT: Before answering any question, ALWAYS use the search_qa_database tool first to check if there's already a stored answer. \
If a matching answer is found in the database, use that answer to ensure consistency. \
If no match is found and you can answer from the provided context, answer normally. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
\n\n\
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n## Resume:\n{self.resume}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def evaluator_system_prompt(self):
        evaluator_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {self.name} and is representing {self.name} on their career website. \
The Agent has been instructed to be professional and engaging, as if talking to a recruiter or future employer who came across the website. \
\n\n\
The Agent has been provided with context on {self.name} in the form of their summary, LinkedIn profile, and resume. \
The Agent also has access to tools including: search_qa_database (must use BEFORE answering), record_unknown_question (when can't answer), and record_user_details (when user provides email). \
\n\n\
Your evaluation should check:\n\
1. Professional and engaging tone appropriate for recruiters/employers\n\
2. Accurate representation of {self.name} based on provided context\n\
3. Staying in character as {self.name}\n\
4. Proper tool usage (especially using search_qa_database before answering, and record_unknown_question when appropriate)\n\
\n\n\
Here's the information about {self.name}:"

        evaluator_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n## Resume:\n{self.resume}\n\n"
        evaluator_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
        return evaluator_prompt

    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(self, reply, message, history) -> Evaluation:
        messages = [
            {"role": "system", "content": self.evaluator_system_prompt()},
            {"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}
        ]
        response = gemini.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=messages,
            response_format=Evaluation
        )
        return response.choices[0].message.parsed

    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]

        # Stream the rerun response
        stream = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            stream=True
        )

        partial_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                partial_reply += chunk.choices[0].delta.content
                yield partial_reply

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        max_retries = 1
        retry_count = 0

        # Phase 1: Handle tool calls (non-streaming)
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message_obj = response.choices[0].message
                tool_calls = message_obj.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True

        # Phase 2: Stream final response
        stream = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )

        partial_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                partial_reply += chunk.choices[0].delta.content
                yield partial_reply

        # Phase 3: Evaluate complete response
        evaluation = self.evaluate(partial_reply, message, history)

        if evaluation.is_acceptable:
            print("✓ Passed evaluation - returning reply")
        else:
            print(f"✗ Failed evaluation: {evaluation.feedback}")
            if retry_count < max_retries:
                print(f"  Retrying... (attempt {retry_count + 1}/{max_retries})")
                # Rerun and stream the retry response
                yield from self.rerun(partial_reply, message, history, evaluation.feedback)
            else:
                print("  Max retries reached - returning response anyway")
                yield partial_reply


if __name__ == "__main__":
    me = Me()

    # Create custom Neo-Memphis Retro Digital theme
    neo_memphis_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.pink,      # Hot pink for interactive elements
        secondary_hue=gr.themes.colors.yellow,  # Sunshine yellow for accents
        neutral_hue=gr.themes.colors.cyan,      # Bright cyan for backgrounds
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_md,
        text_size=gr.themes.sizes.text_md,
        font=gr.themes.GoogleFont("Poppins"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono")
    )

    # Custom CSS for Neo-Memphis styling with optimized spacing
    custom_css = """
    /* Clean background without grid pattern */
    body, .gradio-container {
        background-color: #80D8E8 !important;
    }

    /* Main chat container with reduced border and shadow */
    .contain, .chatbot {
        background: #FFFFFF !important;
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        border-radius: 12px !important;
        margin: 0 !important;
    }

    /* Chat message bubbles with optimized spacing */
    .message-row, .message {
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        background: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 10px !important;
        margin: 4px 0 !important;
    }

    /* User messages with coral/peach accent */
    .user .message, .user.message {
        background: linear-gradient(135deg, #FFE5E5 0%, #FFD4D4 100%) !important;
        border-color: #000000 !important;
    }

    /* Bot messages with white background */
    .bot .message, .bot.message {
        background: #FFFFFF !important;
    }

    /* All buttons with optimized borders */
    button {
        border: 2px solid #000000 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 2px 2px 0px #000000 !important;
        border-radius: 10px !important;
    }

    button:hover {
        transform: translate(-1px, -1px) !important;
        box-shadow: 3px 3px 0px #000000 !important;
    }

    /* Example buttons with sunshine yellow */
    .examples button, button.secondary {
        background: #FFD700 !important;
        color: #000000 !important;
    }

    .examples button:hover, button.secondary:hover {
        background: #FFC700 !important;
    }

    /* Primary buttons (send, submit) with hot pink */
    button.primary, .submit-btn, button[type="submit"] {
        background: #FF1493 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    button.primary:hover, .submit-btn:hover, button[type="submit"]:hover {
        background: #FF69B4 !important;
    }

    /* Input fields and textareas with compact padding */
    input, textarea, .input-field {
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        background: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }

    input:focus, textarea:focus, .input-field:focus {
        border-color: #E6B8FF !important;
        box-shadow: 2px 2px 0px #E6B8FF !important;
        outline: none !important;
    }

    /* Custom scrollbar with bold styling */
    ::-webkit-scrollbar {
        width: 14px;
    }

    ::-webkit-scrollbar-track {
        background: #80D8E8;
        border: 2px solid #000000;
        border-radius: 0px;
    }

    ::-webkit-scrollbar-thumb {
        background: #FF1493;
        border: 2px solid #000000;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #FF69B4;
    }

    /* Headers with subtle text shadow */
    h1, h2, h3, h4 {
        color: #000000 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 0px #FFD700 !important;
    }

    /* Labels and text */
    label {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Flat shadow on all blocks for layered look */
    .form, .block, .panel {
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        background: #FFFFFF !important;
        border-radius: 12px !important;
    }
    """

    demo = gr.ChatInterface(
        me.chat,
        type="messages",
        theme=neo_memphis_theme,
        css=custom_css,
        examples=[
            "What is your professional background?",
            "What are your key skills?",
            "Tell me about your experience",
            "How can I get in touch with you?"
        ]
    )
    demo.launch()
