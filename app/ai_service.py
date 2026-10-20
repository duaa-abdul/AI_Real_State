import os

from groq import Groq
from dotenv import load_dotenv

from app.rag_service import get_relevant_context

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def ask_real_estate_assistant(question):
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    sources = get_relevant_context(question, top_k=3)

    if not sources:
        raise RuntimeError(
            "No knowledge context is available. Check the knowledge base."
        )

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in sources
    )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful real estate educational assistant. "
                    "Answer using the supplied retrieved context. "
                    "If the context does not contain the answer, clearly "
                    "say that the available guide does not specify it. "
                    "Do not invent property listings, prices, legal rules, "
                    "or current market facts. Keep answers clear and concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"User question: {question}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list({
            item["source"] for item in sources
        }),
    }
