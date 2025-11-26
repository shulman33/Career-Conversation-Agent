import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel
from models import Evaluation


# Set up Gemini client for evaluation using OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=GEMINI_BASE_URL
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client
)


def create_evaluator_agent(name: str, summary: str) -> Agent:
    """Create an evaluator agent for the given person."""
    instructions = f"""You are an evaluator that decides whether a response is acceptable quality.
You evaluate responses from an AI agent that is playing the role of {name} on their career website.

The agent should be:
1. Professional and engaging - appropriate tone for recruiters/employers
2. Accurate - only stating facts that are in the provided context
3. In character - responding as {name}, not as a generic AI
4. Helpful - properly using tools like search_qa_database before answering

Context about {name}:
{summary}

Evaluate the response and provide:
- is_acceptable: true if the response meets quality standards, false otherwise
- feedback: specific feedback about what was good or what needs improvement"""

    return Agent(
        name="Response Evaluator",
        instructions=instructions,
        output_type=Evaluation,
        model=gemini_model
    )
