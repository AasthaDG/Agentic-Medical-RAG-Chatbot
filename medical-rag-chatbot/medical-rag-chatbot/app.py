"""
app.py
------
Gradio-based web UI for the Medical RAG Chatbot.

Features:
  - Multi-turn chat with conversation history
  - Source citations shown per response (which medical conversations were retrieved)
  - Intent badge + emergency detection
  - Example questions to get started

Run:
    python app.py

Then open: http://localhost:7860
For a public shareable link: python app.py --share
"""

import argparse
import gradio as gr

from orchestrator import MedicalRAGOrchestrator

# Load RAG system once at startup
print("🚀  Starting Medical RAG Chatbot...")
rag = MedicalRAGOrchestrator.from_disk()
print("✅  Ready\n")


# ── Core chat function ────────────────────────────────────────────────────────

def respond(user_message: str, history: list) -> tuple:
    """
    Main chat handler. Called by Gradio on each user message.

    Args:
        user_message: text from the input box
        history: list of [user_msg, assistant_msg] pairs

    Returns:
        Tuple of (updated_history, sources_markdown, intent_label)
    """
    if not user_message.strip():
        return history, "", ""

    result = rag.answer(user_message)

    # Format the response
    response_text = result["response"]

    # Format source citations
    sources_md = "### 📚 Retrieved Sources\n"
    for src in result["sources"]:
        sources_md += (
            f"**#{src['rank']}** (similarity: `{src['score']}`)\n"
            f"> {src['excerpt']}...\n\n"
        )
    if not result["sources"]:
        sources_md += "_No sources retrieved._"

    # Intent badge
    intent_colors = {
        "symptom_inquiry":   "🔵",
        "medication_question": "🟠",
        "diagnosis_help":    "🟣",
        "treatment_options": "🟢",
        "general_health":    "⚪",
        "emergency":         "🔴",
    }
    icon = intent_colors.get(result["intent"], "⚪")
    intent_label = f"{icon} Intent: **{result['intent'].replace('_', ' ').title()}**"

    history.append([user_message, response_text])
    return history, sources_md, intent_label


def clear_all():
    return [], "", ""


# ── UI Layout ────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "I've had a persistent headache for 3 days with mild fever. What could it be?",
    "Can I take ibuprofen and acetaminophen together?",
    "What are the early warning signs of diabetes?",
    "Is it safe to exercise with a herniated disc?",
    "What causes heart palpitations and when should I be worried?",
    "I feel dizzy when I stand up quickly — what causes this?",
]

CSS = """
.source-box { background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 0.88em; }
.intent-box { font-size: 0.95em; color: #555; padding: 6px 0; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="🏥 Medical RAG Chatbot",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
) as demo:

    # Header
    gr.Markdown("""
    # 🏥 Agentic Medical RAG Chatbot
    **AI-powered assistant grounded in 10,000+ real doctor–patient conversations.**
    
    This system uses a three-stage agentic pipeline:
    `PlannerAgent` → `RetrieverAgent` → `GeneratorAgent`
    
    > ⚠️ This is an AI research tool. Always consult a licensed physician for medical advice.
    """)

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Medical Assistant",
                height=480,
                bubble_full_width=False,
                show_label=True,
            )

            with gr.Row():
                txt_input = gr.Textbox(
                    placeholder="Ask a medical question...",
                    show_label=False,
                    scale=5,
                    lines=1,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)

            clear_btn = gr.Button("🗑 Clear conversation", size="sm", variant="secondary")

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=txt_input,
                label="💡 Example questions",
            )

        # Right: Metadata panel
        with gr.Column(scale=2):
            intent_display = gr.Markdown(
                value="_Intent will appear here after your first message._",
                elem_classes=["intent-box"],
                label="Query Intent",
            )
            sources_display = gr.Markdown(
                value="_Sources will appear here after your first message._",
                elem_classes=["source-box"],
                label="Retrieved Sources",
            )

    # ── How it works ──
    with gr.Accordion("⚙️ How this works", open=False):
        gr.Markdown("""
        **Three-stage agentic pipeline:**

        1. **PlannerAgent** — Classifies your query intent (symptom inquiry, medication question, etc.) 
           and rewrites it for precise semantic search.

        2. **RetrieverAgent** — Embeds the rewritten query using `sentence-transformers/all-MiniLM-L6-v2` 
           and searches a FAISS vector index of 10,000+ doctor–patient conversations for the most relevant context.

        3. **GeneratorAgent** — Sends the retrieved context + your original question to GPT, producing 
           a grounded, citation-backed response.

        **Key components:** FAISS · sentence-transformers · OpenAI GPT · Python
        """)

    # State
    state = gr.State([])

    # Wire events
    send_btn.click(
        fn=respond,
        inputs=[txt_input, state],
        outputs=[chatbot, sources_display, intent_display],
    ).then(lambda: "", outputs=txt_input)

    txt_input.submit(
        fn=respond,
        inputs=[txt_input, state],
        outputs=[chatbot, sources_display, intent_display],
    ).then(lambda: "", outputs=txt_input)

    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, sources_display, intent_display],
    ).then(lambda: [], outputs=state)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
