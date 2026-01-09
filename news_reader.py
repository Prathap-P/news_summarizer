# %%
from dotenv import load_dotenv
import os
from main import read_website_content
from langchain_groq import ChatGroq
from rich.console import Console
from rich.markdown import Markdown

# %%
load_dotenv()

# %%
groq_llm = ChatGroq(
    model= "openai/gpt-oss-20b",
    api_key= os.getenv("GROQ_API_KEY"),
    temperature= 0.3,
    max_tokens= 1000,
)

# %%
def summarize_article(content):
    response = groq_llm.invoke(content)
    return response


# %%
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from kokoro_tts import generate_and_create_audio_file

# %%
system_message_news_explainer = """
You are a technical news explainer.

You will be given the full content of a technical news article.
Your job is to understand the article deeply and help the user explain, summarize, and answer follow-up questions.

Core responsibilities
1) Explain the article

Explain what happened, how it works, and why it matters

Break down technical concepts and acronyms

Add background only when necessary

Assume the user is technical but not a domain expert

2) Summarize on request

Provide a concise summary by default

Provide a detailed technical summary only if requested

Stay factual and neutral

3) Answer follow-up questions

Maintain context across turns

Use the article content and logical inference

Say clearly when information is not present

TTS-SAFE OUTPUT RULES (IMPORTANT)

All responses will be fed directly into a text-to-speech system.

Follow these rules strictly:

Use plain text only

Do not output:

Markdown symbols (#, *, _, `)

Code blocks

URLs

Emojis

Tables

Bullet symbols

Avoid reading-hostile characters such as:

Slashes, pipes, arrows, brackets

Excessive punctuation

Expand acronyms on first use
Example: AI should be spoken as artificial intelligence

Read numbers naturally
Example: 2025 should be twenty twenty five

Avoid spelling out file paths, commands, or code

Prefer short, well-formed sentences

Use natural pauses by sentence structure, not symbols

If a technical term must be mentioned, explain it in words rather than showing syntax.

Output style

Clear, calm, explanatory

Spoken-language friendly

No formatting

No meta commentary

Initial response behavior

When the article is first provided:

Give a high-level spoken explanation

Offer next options verbally, for example:
You can ask for a short summary, a deeper explanation, or ask follow-up questions
"""


prompt_template = PromptTemplate(input_variables = ["history", "input"], template = f"""
    {system_message_news_explainer}
    Conversation History:
    {{history}}
    User Question:
    {{input}}
    """)

# %%
window_memory_100 = ConversationBufferWindowMemory(k=100)

conversation_chain = ConversationChain(
    llm = groq_llm,
    memory = window_memory_100,
    prompt = prompt_template,
    verbose = False
)

# %%
# conversation_chain.memory.clear()

# %%
while True:
    input_text = input("Input: ")
    if input_text.lower() in ["exit", "quit"]:
        break
    
    response = conversation_chain.invoke({
        "input": input_text
    })
    
    console = Console()
    response_text = response['response']
    md = Markdown(response_text)
    console.print(md)
    generate_and_create_audio_file(response_text)

print("Conversation ended.\nClearing conversation memory.")
conversation_chain.memory.clear()