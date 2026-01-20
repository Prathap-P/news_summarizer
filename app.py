import json

from flask import Flask, render_template, request, jsonify, send_file, stream_with_context, Response
from dotenv import load_dotenv
import os
import re
import time

from jupyter_client.session import session_aliases
from langchain.chains.conversation.base import ConversationChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from main import read_website_content
from youtube_transcript_fetcher import get_youtube_transcript
from system_prompts import news_explainer_system_message, youtube_transcript_shortener_system_message
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from kokoro_tts import generate_audio, create_audio_file
import requests

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
    top_p= 0.85,
    max_completion_tokens= -1,
    model_kwargs= {
        "frequency_penalty": 1.3, # Heavily discourages "The speaker says..." loops
        "presence_penalty": 0.3,  # Encourages introducing new topics/facts
    },
    streaming=True,
    stream_usage=True
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
session_history = InMemoryChatMessageHistory()

def check_llm_server():
    """Check if the local LLM server is running"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False


def remove_thinking_tokens(text: str) -> str:
    """
    Remove thinking tokens and reasoning artifacts from LLM response.

    Handles multiple formats:
    - <think>...</think>
    - <|thinking|>...</|thinking|>
    - [thinking]...[/thinking]
    - **Thinking:** ...
    - thinking: ...

    Args:
        text: Raw LLM response text

    Returns:
        Cleaned text with thinking tokens removed
    """
    if not text:
        return text

    original_length = len(text)
    cleaned = text

    # Define patterns to remove (order matters - most specific first)
    patterns = [
        # Paired tags with content
        (r'<\|thinking\|>.*?<\/\|thinking\|>', 'pipe-delimited thinking tags'),
        (r'<think>.*?<\/think>', 'angle bracket thinking tags'),
        (r'\[thinking\].*?\[\/thinking\]', 'square bracket thinking tags'),

        # Markdown-style thinking headers with content until next section
        (r'\*\*[Tt]hinking:?\*\*.*?(?=\n\n|\*\*[A-Z]|$)', 'markdown thinking headers'),

        # Plain text thinking patterns
        (r'(?:^|\n)[Tt]hinking:.*?(?=\n\n|$)', 'plain thinking headers'),

        # Orphaned/malformed tags
        (r'<\/think>', 'orphaned closing think tag'),
        (r'<think>', 'orphaned opening think tag'),
        (r'<\/\|thinking\|>', 'orphaned closing pipe tag'),
        (r'<\|thinking\|>', 'orphaned opening pipe tag'),
        (r'\[\/thinking\]', 'orphaned closing bracket tag'),
        (r'\[thinking\]', 'orphaned opening bracket tag'),
    ]

    # Apply each pattern
    for pattern, description in patterns:
        before_len = len(cleaned)
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        removed = before_len - len(cleaned)
        if removed > 0:
            print(f"[DEBUG] Removed {removed} chars via {description}")

    # Normalize whitespace
    # Remove multiple consecutive newlines (keep max 2)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    # Remove spaces before punctuation
    cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)

    # Log summary
    total_removed = original_length - len(cleaned)
    if total_removed > 0:
        print(f"[CLEANUP] Removed {total_removed} total characters ({(total_removed / original_length) * 100:.1f}%)")
    else:
        print("[CLEANUP] No thinking tokens found")

    return cleaned

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
        System:
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

def create_runnable_chain(mode):
    """Create a runnable chain with mode-specific system prompt"""
    if mode == "news":
        system_message = news_explainer_system_message
    elif mode == "youtube":
        system_message = youtube_transcript_shortener_system_message
    else:
        raise ValueError(f"Invalid mode: {mode}")

    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    base_chain = chat_prompt_template | nemotron_local_llm | StrOutputParser()

    return RunnableWithMessageHistory(
        runnable=base_chain,  # 'runnable' is the required keyword
        get_session_history=lambda session_id: session_history,  # Lambda needs an arg
        input_messages_key="input",
        history_messages_key="history"
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
        conversation_chain = create_runnable_chain(mode)
        current_mode = mode
        global content

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

        # Calculate word count
        word_count = len(content.split())
        
        # Store the content in conversation memory for context
        session_history.add_user_message(f"Here is the {'article' if mode == 'news' else 'video transcript'} content (Total: {word_count} words):\n\n{content}")
        session_history.add_ai_message(f"I have received and analyzed the content. I'm ready to answer your questions about it.")
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
        
        # Track token usage with callback and time
        import time
        llm_start_time = time.time()
        
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
        
        llm_time = time.time() - llm_start_time
        response_text = response['response']
        
        # Remove thinking tokens from response
        response_text = remove_thinking_tokens(response_text)
        
        print(f"[SUCCESS] LLM response generated. Length: {len(response_text)} characters")
        print(f"[TOKENS] Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
        print(f"[TIME] LLM generation took: {llm_time:.2f}s")
        
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
        audio_time = None

        # Generate audio if requested
        if generate_audio_flag:
            try:
                print(f"[INFO] Generating audio for response ({len(response_text)} chars)...")
                audio_start_time = time.time()
                
                audio = generate_audio(response_text)
                audio_file_path = create_audio_file(audio)
                audio_file = os.path.basename(audio_file_path)
                
                audio_time = time.time() - audio_start_time
                print(f"[SUCCESS] Audio generated: {audio_file}")
                print(f"[TIME] Audio generation took: {audio_time:.2f}s")
            except Exception as e:
                print(f"[ERROR] Audio generation failed: {e}")

        return jsonify({
            'response': response_text,
            'audio_file': audio_file,
            'token_usage': token_usage,
            'llm_time': round(llm_time, 2),
            'audio_time': round(audio_time, 2) if audio_time else None,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/streamChat', methods=['POST'])
def stream_chat():
    """Handle follow-up questions with streaming"""
    # Check if LLM server is accessible first
    if not check_llm_server():
        return jsonify({
            'error': 'Local LLM server is not running. Please start LM Studio and load a model at http://localhost:1234'
        }), 503

    data = request.json
    user_input = data.get('message')
    generate_audio_flag = data.get('generate_audio', False)

    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    def stream_response():
        try:
            # Track token usage with callback and time
            import time
            
            token_usage = None
            audio_file = None
            audio_time = None

            llm_start_time = time.time()
            total_chunk_size = 0
            complete_response_text = ""

            print(f"[INFO] Generating streaming LLM response for: '{user_input[:50]}...'")

            with get_openai_callback() as cb:
                chunk_received = False
                # with get_openai_callback() as cb:
                for chunk_response in conversation_chain.stream({
                    "input": user_input
                }, config = {"configurable": {"session_id": "any_string_here"}}):

                    if chunk_response:  # Only send non-empty chunks
                        chunk_received = True
                        complete_response_text += chunk_response
                        total_chunk_size += len(chunk_response)
                        # Send only the chunk during streaming
                        chunk_data = {'chunk': chunk_response}
                        yield f"data:{json.dumps(chunk_data)}\n\n"

                if not chunk_received:
                    raise Exception("No response from LLM - check if local LLM server is running")

                # Extract token usage from callback
                token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens
                }

            llm_time = time.time() - llm_start_time

            # Remove thinking tokens from complete response
            complete_response_text = remove_thinking_tokens(complete_response_text)

            # print(f"[SUCCESS] LLM streaming response completed. Length: {total_chunk_size} characters")
            # print(f"[TOKENS] Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")
            # print(f"[TIME] LLM generation took: {llm_time:.2f}s")

            # Send streaming-done event if audio generation is requested
            if generate_audio_flag:
                streaming_done_data = {'streaming_done': True}
                yield f"data:{json.dumps(streaming_done_data)}\n\n"

            # Generate audio if requested
            if generate_audio_flag:
                try:
                    print(f"[INFO] Generating audio for response ({len(complete_response_text)} chars)...")
                    audio_start_time = time.time()

                    audio = generate_audio(complete_response_text)
                    audio_file_path = create_audio_file(audio)
                    audio_file = os.path.basename(audio_file_path)

                    audio_time = time.time() - audio_start_time
                    print(f"[SUCCESS] Audio generated: {audio_file}")
                    print(f"[TIME] Audio generation took: {audio_time:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Audio generation failed: {e}")

            # Send final metadata message
            final_data = {
                'done': True,
                'audio_file': audio_file,
                'token_usage': token_usage,
                'llm_time': round(llm_time, 2),
                'audio_time': round(audio_time, 2) if audio_time else None,
                'success': True
            }

            yield f"data:{json.dumps(final_data)}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Stream chat error: {e}")
            error_data = {'error': str(e), 'done': True, 'success': False}
            yield f"data:{json.dumps(error_data)}\n\n"
    
    return Response(stream_with_context(stream_response()),
                    mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no',
                         'Cache-Control': 'no-cache'
                    })


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
    
    if session_history:
        session_history.clear()
    
    conversation_chain = None
    current_mode = None

    return jsonify({'success': True, 'message': 'Conversation cleared'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)