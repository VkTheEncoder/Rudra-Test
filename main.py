# main.py

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from langchain_ollama.llms import OllamaLLM
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

from vector import retriever  # your Chroma-backed retriever

# 🔐 API key
API_KEY = "rudra-ai-123456"

# 🚀 FastAPI app
app = FastAPI()

# ✅ Request schema
class Query(BaseModel):
    question: str

# 🤖 Load your local LLaMA model
model = OllamaLLM(model="llama3.2:1b")

# 🧠 Prompt template
template = """
You are a friendly AI mentor helping a 10-year-old mindset who has no experience with coding, computers, or technology.
Only answer *exactly* what the user has asked. Do *not add any extra or unrelated information*.
Use very simple words and explain slowly like you're talking to a curious child. Now write the answer in short, simple sentences. Use analogies and real-life examples when possible. Keep it relevant to the context. Avoid guessing if unsure.

Always:
- do not reply with kiddo or little friend!
- your target audience is younger but not aware of technologies 
- Start with a kind greeting or encouragement
- Explain using small examples or analogies
- Avoid technical words unless you explain them clearly
- Be warm, fun, and supportive

If possible, answer in the student's local language if the context shows it.

If the context does not contain enough information to answer the question, simply say:
*"I'm not sure about that yet. Please ask something related to coding or digital skills."*

Context:
{context}

Question:
{question}

Now explain the answer in the easiest way possible.
"""

# 🔗 Build the LLMChain
prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=model, prompt=prompt)

# 🌐 API endpoint
@app.post("/ask")
async def ask_ai(query: Query, x_api_key: str = Header(...)):
    # 🔐 API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🧠 Retrieval
    try:
        docs = retriever.get_relevant_documents(query.question)
        context = "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        print("⚠ Retriever error:", e)
        context = "No helpful information was found in the database."

    # 🤖 Generation
    try:
        answer_text = chain.run(context=context, question=query.question)
        return {
            "answer": f"Namaste 👋. Let's talk about coding.\n\n{answer_text}"
        }
    except Exception as e:
        print("❌ Generation error:", e)
        raise HTTPException(status_code=500, detail="Error generating answer")

# 🖥 CLI for local testing
if __name__ == "__main__":
    print("🔵 Welcome to Mentor AI!")
    print("Ask anything about coding, digital skills, or job-related training.")
    print("Type 'q' to quit.\n")

    while True:
        question = input("❓ Ask your question: ").strip()
        if question.lower() == "q":
            print("👋 Goodbye! Keep learning.")
            break

        try:
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            print("⚠ Retriever error:", e)
            context = "No helpful information was found in the database."

        try:
            reply = chain.run(context=context, question=question)
            print(f"\n🤖 AI:\n{reply}\n")
        except Exception as e:
            print("❌ Generation error:", e)
