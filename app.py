from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import os
import re
from main import read_website_content
from youtube_transcript_fetcher import get_youtube_transcript
from system_prompts import news_explainer_system_message, youtube_transcript_shortener_system_message
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from kokoro_tts import generate_audio, create_audio_file

load_dotenv()

app = Flask(__name__)

# Initialize LLM
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_completion_tokens=65000,
)

gemma_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.7,
    model="google/gemma-3-27b"
)

nemotron_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.3,
    model="nvidia/nemotron-3-nano",
    top_p= 0.6,
    max_completion_tokens= -1,
    model_kwargs= {
        "frequency_penalty": 0.5, # Heavily discourages "The speaker says..." loops
        "presence_penalty": 0.3,  # Encourages introducing new topics/facts
    }
)

deepseekR1_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.7,
    model="deepseek/deepseek-r1-0528-qwen3-8b"
)

gpt_oss_20b_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="openai/gpt-oss-20b",
    extra_body={"reasoning_effort": "high"},
    max_completion_tokens=12800,
    temperature=0.5,
    # streaming=True,
)

# Global conversation state (in production, use session-based storage)
window_memory_100 = ConversationBufferWindowMemory(k=100)
conversation_chain = None
current_mode = None


def remove_thinking_tokens(text):
    """Remove thinking tokens and related patterns from LLM response"""
    # Remove <think>...</think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove orphaned closing </think> tags
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
    # Remove orphaned opening <think> tags
    text = re.sub(r'<think>', '', text, flags=re.IGNORECASE)
    # Remove thinking: ... patterns
    text = re.sub(r'thinking:.*?(?=\n\n|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove [thinking] ... [/thinking] patterns
    text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()


def create_conversation_chain(mode):
    """Create a conversation chain with mode-specific system prompt"""
    if mode == "news":
        system_message = news_explainer_system_message
    elif mode == "youtube":
        system_message = youtube_transcript_shortener_system_message
    else:
        raise ValueError(f"Invalid mode: {mode}")

    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""
        {system_message}
        Conversation History:
        {{history}}
        User Question:
        {{input}}
        """
    )

    return ConversationChain(
        llm=nemotron_local_llm,
        memory=window_memory_100,
        prompt=prompt_template,
        verbose=False
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_content', methods=['POST'])
def load_content():
    """Load content (news article or YouTube transcript) and return raw content"""
    global conversation_chain, current_mode

    data = request.json
    url = data.get('url')
    mode = data.get('mode', 'news')  # 'news' or 'youtube'

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    if mode not in ['news', 'youtube']:
        return jsonify({'error': 'Invalid mode. Use "news" or "youtube"'}), 400

    try:
        # Initialize conversation chain for the selected mode
        conversation_chain = create_conversation_chain(mode)
        current_mode = mode

        if mode == 'news':
            # Load news article content
            print(f"[INFO] Loading news article from: {url}")
            documents = read_website_content(url)
            if not documents:
                print(f"[ERROR] Failed to load article from: {url}")
                return jsonify({'error': 'Could not load article from URL'}), 400

            content = documents[0].page_content
            print(f"[SUCCESS] Article loaded successfully. Length: {len(content)} characters")

        elif mode == 'youtube':
            # Get YouTube transcript
            print(f"[INFO] Fetching YouTube transcript from: {url}")
            content = get_youtube_transcript(url)

            # Check if transcript fetch failed
            if content.startswith("No transcript available") or content.startswith("Error fetching") or content.startswith("Invalid YouTube"):
                print(f"[ERROR] YouTube transcript fetch failed: {content}")
                return jsonify({
                    'error': content,
                    'success': False
                }), 400
            
            print(f"[SUCCESS] YouTube transcript fetched successfully. Length: {len(content)} characters")

        # Store the content in conversation memory for context
        conversation_chain.memory.save_context(
            {"input": f"Here is the {'article' if mode == 'news' else 'video transcript'} content:\n\n{content}"},
            {"output": "I have received and analyzed the content. I'm ready to answer your questions about it."}
        )

        # Calculate word count
        word_count = len(content.split())
        print(f"[INFO] Word count: {word_count}")

        # Return the raw content to display to user
        return jsonify({
            'content': content,
            'mode': mode,
            'word_count': word_count,
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
        print(f"[INFO] Generating LLM response for: '{user_input[:50]}...'")
        
        # Track token usage with callback
        with get_openai_callback() as cb:
            response = conversation_chain.invoke({
                "input": user_input
            })
            
            # Extract token usage from callback
            token_usage = {
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_tokens': cb.total_tokens
            }

        response_text = response['response']
        
        # Remove thinking tokens from response
        response_text = remove_thinking_tokens(response_text)
        
        print(f"[SUCCESS] LLM response generated. Length: {len(response_text)} characters")
        print(f"[TOKENS] Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
        
        # Get token usage if available
        token_usage = None
        # if hasattr(groq_llm, 'last_response') and groq_llm.last_response:
        #     usage = groq_llm.last_response.get('usage', {})
        #     if usage:
        #         token_usage = {
        #             'prompt_tokens': usage.get('prompt_tokens', 0),
        #             'completion_tokens': usage.get('completion_tokens', 0),
        #             'total_tokens': usage.get('total_tokens', 0)
        #         }
        
        audio_file = None

        # Generate audio if requested
        if generate_audio_flag:
            try:
                print(f"[INFO] Generating audio for response ({len(response_text)} chars)...")
                audio = generate_audio(response_text)
                audio_file_path = create_audio_file(audio)
                audio_file = os.path.basename(audio_file_path)
                print(f"[SUCCESS] Audio generated: {audio_file}")
            except Exception as e:
                print(f"[ERROR] Audio generation failed: {e}")

        return jsonify({
            'response': response_text,
            'audio_file': audio_file,
            'token_usage': token_usage,
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
    """Clear conversation memory and reset mode"""
    global conversation_chain, current_mode
    
    if conversation_chain:
        conversation_chain.memory.clear()
    
    conversation_chain = None
    current_mode = None
    
    return jsonify({'success': True, 'message': 'Conversation cleared'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)