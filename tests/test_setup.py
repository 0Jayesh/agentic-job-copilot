import __init__ 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
response = llm.invoke("Say hello in 5 words or less.")
for block in response.content:
    if isinstance(block, dict) and block.get("type") == "text":
        print(block["text"])