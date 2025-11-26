from pydantic import BaseModel, Field
from typing import List


class Evaluation(BaseModel):
    """Model for response evaluation results."""
    is_acceptable: bool
    feedback: str


class ChatSummary(BaseModel):
    """Model for email chat summary."""
    user_name: str = Field(description="User's name if provided, otherwise 'Visitor'")
    user_email: str = Field(description="User's email address")
    topics_discussed: List[str] = Field(description="Main topics covered in conversation")
    user_interests: str = Field(description="What the user seems interested in")
    conversation_sentiment: str = Field(description="Overall tone: positive, neutral, curious, etc.")
    notable_questions: List[str] = Field(description="Key questions the user asked")


class InappropriateContentOutput(BaseModel):
    """Output model for inappropriate content guardrail."""
    is_inappropriate: bool
    reasoning: str


class PromptInjectionOutput(BaseModel):
    """Output model for prompt injection guardrail."""
    is_injection_attempt: bool
    reasoning: str


class OffTopicOutput(BaseModel):
    """Output model for off-topic guardrail."""
    is_off_topic: bool
    reasoning: str


class CompetitorMentionOutput(BaseModel):
    """Output model for competitor mention guardrail."""
    mentions_competitor: bool
    competitor_names: List[str] = Field(default_factory=list)
