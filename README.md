---
title: career_conversation
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---

# AI Career Assistant

An intelligent, agentic chatbot that represents me to recruiters and employers on my personal website. Built with the OpenAI Agents SDK, this project showcases modern AI agent architecture patterns.

## What It Does

When recruiters visit my website, they interact with an AI agent that:
- Answers questions about my background, skills, and experience
- Maintains consistent, accurate responses via a semantic Q&A database
- Proactively offers to connect visitors with me via email
- Self-evaluates response quality using a secondary AI model
- Protects against prompt injection and inappropriate content

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Frontend                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Career Manager                           │
│                   (Orchestration Layer)                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────────┐
│  Input          │  │   Main      │  │  Evaluator Agent    │
│  Guardrails     │  │   Agent     │  │  (Gemini 2.0 Flash) │
│  (4 checks)     │  │ (GPT-4o-mini)│  │                     │
└─────────────────┘  └─────────────┘  └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────────┐
│  Q&A Database   │  │   Email     │  │  Unknown Questions  │
│  (SQLite)       │  │   Tool      │  │  Tracking           │
└─────────────────┘  └─────────────┘  └─────────────────────┘
```

## Key Techniques

### 1. Agent-as-Tool Pattern
The evaluator agent is converted into a tool using `agent.as_tool()`, allowing the main agent to optionally self-evaluate responses before sending them. This creates a hierarchical multi-agent system within a single conversation.

```python
evaluator_agent = create_evaluator_agent(name, summary)
evaluator_tool = evaluator_agent.as_tool(
    tool_name="evaluate_my_response",
    tool_description="Self-evaluate your response quality..."
)
```

### 2. Dual-Model Architecture
- **Main Agent**: GPT-4o-mini for fast, cost-effective responses
- **Evaluator Agent**: Gemini 2.0 Flash for quality assessment (different provider, different perspective)

### 3. Semantic Q&A Search
Rather than keyword matching, the agent uses GPT-4o-mini to semantically match user questions against a knowledge base, ensuring consistent answers even when questions are phrased differently.

### 4. Input Guardrails
Four-layer protection system:
| Guardrail | Type | Purpose |
|-----------|------|---------|
| Inappropriate Content | Blocking | Filters profanity, harassment, unprofessional content |
| Prompt Injection | Blocking | Detects jailbreak attempts, instruction overrides |
| Off-Topic | Soft | Logs when conversations drift from career topics |
| Competitor Mention | Soft | Analytics for tracking recruiter sources |

### 5. Unknown Question Tracking
Smart handling of unanswered questions:
- Separate table prevents knowledge base pollution
- Automatic deduplication with frequency counting
- Dismissal workflow for irrelevant questions
- Promotion workflow to move answered questions to main database

### 6. Proactive Engagement
The agent is instructed to naturally suggest email contact after building rapport, converting interested visitors into real conversations.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | OpenAI Agents SDK |
| Primary LLM | GPT-4o-mini |
| Evaluator LLM | Gemini 2.0 Flash |
| Frontend | Gradio |
| Database | SQLite |
| Hosting | Hugging Face Spaces |

## Project Structure

```
├── app.py                 # Gradio application entry point
├── career_manager.py      # Orchestration layer
├── career_agents/
│   ├── career_agent.py    # Main agent with tools
│   ├── evaluator_agent.py # Quality evaluation agent
│   ├── email_agent.py     # Email notification tool
│   └── guardrails.py      # Input protection layer
├── tools/
│   └── qa_tools.py        # Database interaction tools
├── database/
│   └── qa_database.py     # SQLite operations
├── models/
│   └── *.py               # Pydantic models for structured outputs
└── me/
    ├── summary.md         # Q&A knowledge base (markdown)
    ├── resume.pdf         # Resume for context
    └── linkedin.pdf       # LinkedIn profile for context
```

## Local Development

```bash
# Clone and setup
git clone https://github.com/shulman33/career-conversation.git
cd career-conversation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your API keys: OPENAI_API_KEY, GOOGLE_API_KEY

# Run
python app.py
```

## What I Learned

Building this project taught me:
- How to compose agents using the agent-as-tool pattern
- Implementing guardrails for production AI systems
- Balancing instruction specificity (too vague = ignored, too strict = brittle)
- Designing databases that separate clean data from analytics data
- The importance of semantic search over keyword matching for Q&A systems

## Live Demo

Try it out: [samjshulman.com](https://www.samjshulman.com)

---

Built by Sam Shulman | [LinkedIn](https://linkedin.com/in/sam-shulman) | [GitHub](https://github.com/shulman33)
