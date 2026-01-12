from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import os
from main import read_website_content
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from kokoro_tts import generate_and_create_audio_file, generate_audio, create_audio_file
import threading

load_dotenv()

app = Flask(__name__)

# Initialize LLM
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1000,
)

# System message for TTS-safe output
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

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=f"""
    {system_message_news_explainer}
    Conversation History:
    {{history}}
    User Question:
    {{input}}
    """
)

# Global conversation chain (in production, use session-based storage)
window_memory_100 = ConversationBufferWindowMemory(k=100)
conversation_chain = ConversationChain(
    llm=groq_llm,
    memory=window_memory_100,
    prompt=prompt_template,
    verbose=False
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_article', methods=['POST'])
def load_article():
    """Load article from URL and provide initial explanation"""
    data = request.json
    url = data.get('url')
    generate_audio_flag = data.get('generate_audio', False)
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Load article content
        documents = read_website_content(url)
        if not documents:
            return jsonify({'error': 'Could not load article from URL'}), 400
        
        article_content = documents[0].page_content
        
        # Get initial explanation from LLM
        response = conversation_chain.invoke({
            "input": f"Here is the article content:\n\n{article_content}"
        })
        
        response_text = response['response']
        audio_file = None
        
        # Generate audio if requested
        if generate_audio_flag:
            try:
                audio = generate_audio(response_text)
                audio_file_path = create_audio_file(audio)
                audio_file = os.path.basename(audio_file_path)
            except Exception as e:
                print(f"Error generating audio: {e}")
        
        return jsonify({
            'response': response_text,
            'audio_file': audio_file,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handle follow-up questions"""
    data = request.json
    user_input = data.get('message')
    generate_audio_flag = data.get('generate_audio', False)
    
    if not user_input:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        # Get response from conversation chain
        response = conversation_chain.invoke({
            "input": user_input
        })
        
        response_text = response['response']
        audio_file = None
        
        # Generate audio if requested
        if generate_audio_flag:
            try:
                audio = generate_audio(response_text)
                audio_file_path = create_audio_file(audio)
                audio_file = os.path.basename(audio_file_path)
            except Exception as e:
                print(f"Error generating audio: {e}")
        
        return jsonify({
            'response': response_text,
            'audio_file': audio_file,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve a specific audio file"""
    audio_dir = 'kokoro_outputs'
    file_path = os.path.join(audio_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav')
    
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation memory"""
    conversation_chain.memory.clear()
    return jsonify({'success': True, 'message': 'Conversation cleared'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
