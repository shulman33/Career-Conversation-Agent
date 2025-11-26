import json
from openai import OpenAI
from agents import function_tool
from database import fetch_all_qa, insert_qa, update_qa


@function_tool
def record_unknown_question(question: str) -> dict:
    """Record any question that couldn't be answered because the answer is not known.
    Use this when you don't have information to answer a user's question.

    Args:
        question: The question that couldn't be answered
    """
    placeholder_answer = "ANSWER NEEDED - Please update this entry in the database"
    insert_qa(question, placeholder_answer)
    print(f"Question needs answer: {question}")
    return {"recorded": "ok", "added_to_database": True, "message": "Question recorded for Sam to answer later"}


@function_tool
def search_qa_database(question: str) -> dict:
    """Search the Q&A database for semantically similar questions.
    Use this BEFORE answering any question to check if there's already a stored answer.
    This helps provide consistent, accurate responses.

    Args:
        question: The user's question to search for in the database
    """
    qa_pairs = fetch_all_qa()

    if not qa_pairs:
        return {"found": False, "answer": None, "message": "Database is empty"}

    # Filter out questions with placeholder answers
    answered_qa_pairs = [qa for qa in qa_pairs if "ANSWER NEEDED" not in qa['answer']]

    if not answered_qa_pairs:
        return {"found": False, "answer": None, "message": "No answered questions in database yet"}

    context = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answered_qa_pairs])

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


@function_tool
def add_qa_to_database(question: str, answer: str) -> dict:
    """Add a new question and answer pair to the database.
    Use this to store commonly asked questions and their answers for future reference.

    Args:
        question: The question to store
        answer: The answer to the question
    """
    insert_qa(question, answer)
    return {"added": True, "message": "Successfully added Q&A pair to database"}


@function_tool
def list_recent_qa(limit: int = 5) -> dict:
    """List recent Q&A pairs from the database.
    Useful for showing what questions have been answered before.

    Args:
        limit: Maximum number of Q&A pairs to retrieve (default: 5)
    """
    qa_pairs = fetch_all_qa()
    recent = qa_pairs[:limit]

    for qa in recent:
        qa['needs_answer'] = "ANSWER NEEDED" in qa['answer']

    return {
        "count": len(recent),
        "qa_pairs": recent,
        "message": f"Retrieved {len(recent)} recent Q&A pairs"
    }


@function_tool
def update_qa_answer(question: str, new_answer: str) -> dict:
    """Update the answer for an existing question in the database.
    Useful for replacing placeholder answers or correcting existing answers.

    Args:
        question: The exact question text to update
        new_answer: The new answer to store for this question
    """
    updated = update_qa(question, new_answer)

    if updated:
        return {"updated": True, "message": f"Successfully updated answer for: {question}"}
    else:
        return {"updated": False, "message": f"Question not found: {question}"}
