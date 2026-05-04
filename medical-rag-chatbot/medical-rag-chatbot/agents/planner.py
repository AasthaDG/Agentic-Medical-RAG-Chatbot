"""
agents/planner.py
-----------------
PlannerAgent — the first stage of the agentic pipeline.

Responsibilities:
  - Classify the user's query intent
  - Rewrite the query to be more clinically precise (query expansion)
  - Detect emergency situations and flag them

This improves retrieval quality downstream because vague patient questions
("my head hurts") get rewritten into precise search queries
("persistent headache with photophobia, possible migraine or tension headache").
"""

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PLANNER, TEMPERATURE_PLANNER

# Intent taxonomy — maps to different retrieval and generation strategies
INTENT_LABELS = [
    "symptom_inquiry",       # patient describing symptoms
    "medication_question",   # drug interactions, dosage, side effects
    "diagnosis_help",        # asking about a specific condition
    "treatment_options",     # what can be done about X
    "general_health",        # lifestyle, prevention, wellness
    "emergency",             # urgent / life-threatening symptoms
]


class PlannerAgent:
    """
    Analyzes and reformulates user queries before retrieval.

    The planner uses a lightweight LLM call to:
      1. Classify intent (determines how urgent / what type of answer is needed)
      2. Rewrite the query for semantic search (improves retrieval precision)

    This is the 'plan' step in the planner-retriever-generator architecture.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "PlannerAgent"

    def plan(self, user_query: str) -> dict:
        """
        Classifies and rewrites the user query.

        Args:
            user_query: Raw patient question

        Returns:
            dict with keys:
              - original_query: unchanged input
              - intent: one of INTENT_LABELS
              - search_query: rewritten query for FAISS retrieval
              - is_emergency: bool flag
        """
        prompt = f"""You are a clinical query analyzer for a medical AI system.

Given the patient's question, do three things:
1. Classify intent as one of: {", ".join(INTENT_LABELS)}
2. Rewrite the query to be clinically precise and specific for searching a medical database.
   Expand abbreviations, add medical terminology, make it search-optimized.
3. Flag if this sounds like a medical emergency (true/false).

Patient question: {user_query}

Respond in EXACTLY this format (no extra text):
INTENT: <intent>
SEARCH_QUERY: <rewritten query>
EMERGENCY: <true or false>"""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_PLANNER,
                max_tokens=MAX_TOKENS_PLANNER,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_response(user_query, text)

        except Exception as e:
            # Fallback plan — never crash the pipeline
            print(f"  [PlannerAgent] ⚠️  LLM call failed: {e}. Using fallback.")
            return {
                "original_query": user_query,
                "intent": "general_health",
                "search_query": user_query,
                "is_emergency": False,
            }

    def _parse_response(self, original_query: str, text: str) -> dict:
        """Parses structured LLM output into a clean dict."""
        lines = {
            line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
            for line in text.splitlines()
            if ":" in line
        }

        intent = lines.get("INTENT", "general_health").lower()
        if intent not in INTENT_LABELS:
            intent = "general_health"

        search_query = lines.get("SEARCH_QUERY", original_query)
        is_emergency = lines.get("EMERGENCY", "false").lower() == "true"

        plan = {
            "original_query": original_query,
            "intent": intent,
            "search_query": search_query,
            "is_emergency": is_emergency,
        }

        print(f"  [PlannerAgent] intent={intent} | emergency={is_emergency}")
        print(f"  [PlannerAgent] search_query: {search_query[:80]}...")
        return plan
