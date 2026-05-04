"""
agents/generator.py
-------------------
GeneratorAgent — produces the final grounded medical response.

Responsibilities:
  - Formats retrieved documents as structured context
  - Applies prompt engineering to ensure grounded, safe, relevant responses
  - Handles emergency routing with appropriate messaging
  - Returns both the response and source citations

This is where retrieval meets generation — the 'G' in RAG.
"""

from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS_GENERATOR,
    TEMPERATURE_GENERATOR,
)

EMERGENCY_MESSAGE = (
    "⚠️ **This sounds like it could be a medical emergency.** "
    "Please call 911 or go to your nearest emergency room immediately. "
    "Do not wait for an AI response in an emergency situation.\n\n"
)

SYSTEM_PROMPT = """You are a helpful, empathetic medical AI assistant grounded in real doctor-patient conversations.

Your job:
- Answer the patient's question using ONLY the provided medical references
- Be factual, clear, and compassionate
- Always recommend consulting a licensed doctor for diagnosis or treatment decisions
- If the references don't match the question well, say so honestly instead of guessing
- Never diagnose — provide information, not diagnosis

Format your response in 2-3 paragraphs. Be human, not robotic."""


class GeneratorAgent:
    """
    Generates grounded, context-aware medical responses.

    Uses retrieved documents as context and applies prompt engineering
    to produce safe, relevant, and empathetic answers.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "GeneratorAgent"

    def generate(self, plan: dict, retrieved_docs: list[dict]) -> dict:
        """
        Generates a response grounded in retrieved medical conversations.

        Args:
            plan: Output from PlannerAgent (intent, original_query, is_emergency)
            retrieved_docs: Output from RetrieverAgent (list of doc dicts)

        Returns:
            dict with:
              - response: the generated answer text
              - sources: list of source excerpts shown to user
              - is_emergency: bool
              - intent: query intent label
        """
        # Handle emergencies immediately
        prefix = EMERGENCY_MESSAGE if plan.get("is_emergency") else ""

        # Format context — use top 3 docs, truncated
        context_blocks = []
        sources = []
        for i, doc in enumerate(retrieved_docs[:3], start=1):
            snippet = doc["document"][:700]
            context_blocks.append(f"[Reference {i}] (similarity: {doc['similarity_score']:.2f})\n{snippet}")
            # Show patient question as the source citation
            patient_line = doc["document"].split("\n")[0].replace("Patient: ", "")
            sources.append({
                "rank": i,
                "excerpt": patient_line[:120],
                "score": round(doc["similarity_score"], 3),
            })

        context = "\n\n".join(context_blocks)

        user_prompt = f"""Patient's question: {plan["original_query"]}
Query intent: {plan["intent"]}

Medical references retrieved from 10,000+ doctor-patient conversations:
{context}

Please answer the patient's question based on these references."""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE_GENERATOR,
                max_tokens=MAX_TOKENS_GENERATOR,
            )
            answer = response.choices[0].message.content.strip()
            full_response = prefix + answer

            print(f"  [GeneratorAgent] Response: {len(full_response)} chars")
            return {
                "response": full_response,
                "sources": sources,
                "is_emergency": plan.get("is_emergency", False),
                "intent": plan.get("intent", "general_health"),
            }

        except Exception as e:
            print(f"  [GeneratorAgent] ⚠️  Generation failed: {e}")
            return {
                "response": "I'm sorry, I encountered an error generating a response. Please try again.",
                "sources": [],
                "is_emergency": False,
                "intent": plan.get("intent", "general_health"),
            }
