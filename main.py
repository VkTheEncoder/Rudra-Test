from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Import your retriever here (Chroma vector DB)

# 🔐 Set your API key here
API_KEY = "rudra-ai-123456"

# 🚀 Create the FastAPI app
app = FastAPI()

# ✅ Define request format
class Query(BaseModel):
    question: str

# 🤖 Load the LLaMA model
model = OllamaLLM(model="llama3.2:1b")  # Make sure model is available locally

# 🧠 Define the prompt template
template = """
You are a friendly AI mentor helping a 10-year-old mindset who has no experience with coding , computers, or technology.
Only answer **exactly** what the user has asked. Do **not add any extra or unrelated information**.
Use very simple words and explain slowly like you're talking to a curious child. Now write the answer in short, simple sentences. Use analogies and real-life examples when possible. Keep it relevant to the context. Avoid guessing if unsure.

Always:
- do not reply with kiddo or little friend
- your targeted audience are good age but they dont aware about technologies 
- Start with a kind greeting or encouragement
- Explain using small examples or analogies
- Avoid technical words unless you explain them clearly
- Be warm, fun, and supportive

If possible, answer in the student's local language if the context shows it.

If the context does not contain enough information to answer the question, simply say:
**"I'm not sure about that yet. Please ask something related to coding or digital skills."**

Context:
{context}

Question:
{question}

Now explain the answer in the easiest way possible.
"""

# 🔗 Create chain (prompt -> model)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 🌐 API endpoint for your AI
@app.post("/ask")
async def ask_ai(query: Query, x_api_key: str = Header(...)):
    # 🔐 Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🧠 Retrieve context
    try:
        docs = retriever.invoke(query.question)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        context = "No helpful information was found in the database."
        print("⚠️ Retriever error:", e)

    # 🤖 Generate answer
    try:
        result = chain.invoke({"context": context, "question": query.question})
        return {"answer": f"Namaste 👋. Let's talk about coding.\n\n{result}"}
    except Exception as e:
        print("❌ Error during response generation:", e)
        return {"answer": "Sorry, I had trouble generating the answer."}


    
