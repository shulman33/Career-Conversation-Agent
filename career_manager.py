from pypdf import PdfReader
from agents import Runner, trace, InputGuardrailTripwireTriggered
from openai.types.responses import ResponseTextDeltaEvent
from database import seed_database
from career_agents import create_career_agent


class CareerManager:
    """Orchestrates the career conversation chatbot."""

    def __init__(self):
        self.name = "Sam Shulman"

        # Load PDF content
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

        # Create the main career agent
        self.agent = create_career_agent(
            name=self.name,
            summary=self.summary,
            linkedin=self.linkedin,
            resume=self.resume
        )

    async def run(self, message: str, history: list):
        """Run the career conversation, yielding status updates and responses."""
        # Add user message to history and yield immediately
        history = history + [{"role": "user", "content": message}]
        yield history

        # Build input for the agent
        input_items = [{"role": m["role"], "content": m["content"]} for m in history]

        try:
            with trace("Career Conversation"):
                # Run the agent with streaming
                result = Runner.run_streamed(
                    self.agent,
                    input=input_items,
                )

                # Add assistant message placeholder with thinking indicator
                history = history + [{"role": "assistant", "content": "..."}]
                yield history

                first_token = True
                async for event in result.stream_events():
                    if event.type == "raw_response_event":
                        if isinstance(event.data, ResponseTextDeltaEvent):
                            if first_token:
                                history[-1]["content"] = event.data.delta
                                first_token = False
                            else:
                                history[-1]["content"] += event.data.delta
                            yield history

        except InputGuardrailTripwireTriggered as e:
            print(f"Guardrail triggered: {e}")
            history = history + [{"role": "assistant", "content": "I'd be happy to discuss Sam's career, skills, and experience. What would you like to know?"}]
            yield history
