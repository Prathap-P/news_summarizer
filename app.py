import json

from flask import Flask, render_template, request, jsonify, send_file, stream_with_context, Response
import os

from langchain.chains.conversation.base import ConversationChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
import requests
import time

from dotenv import load_dotenv
load_dotenv()

from main import read_website_content
from youtube_transcript_fetcher import get_youtube_transcript
from system_prompts import news_explainer_system_message, subject_matter_expert_prompt
from condenser_service import condense_content
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from kokoro_tts import generate_audio, create_audio_file
from llm_models import get_model
from utils import remove_thinking_tokens, create_backup_file
from email_sender import send_email_with_audio, send_email_with_attachments
from telegram_sender import send_telegram_with_audio, send_telegram_with_attachments


app = Flask(__name__)

# Global conversation state (in production, use session-based storage)
window_memory_100 = ConversationBufferWindowMemory(k=100)
conversation_chain = None
current_mode = None
session_history = InMemoryChatMessageHistory()
current_model = get_model("nemotron_stream_local_llm")

def check_llm_server():
    """Check if the local LLM server is running"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_conversation_chain(mode):
    """Create a conversation chain with mode-specific system prompt"""
    if mode == "news":
        system_message = news_explainer_system_message
    elif mode == "youtube":
        system_message = subject_matter_expert_prompt
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
        llm=current_model,
        memory=window_memory_100,
        prompt=prompt_template,
        verbose=False
    )

def create_runnable_chain(mode):
    """Create a runnable chain with mode-specific system prompt"""
    print(f"[INFO] Creating runnable chain for mode: {mode}")
    if mode == "news":
        system_message = news_explainer_system_message
    elif mode == "youtube":
        system_message = subject_matter_expert_prompt
    else:
        raise ValueError(f"Invalid mode: {mode}")

    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    base_chain = chat_prompt_template | current_model | StrOutputParser()

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
    """Load content, condense it using condenser_service, and prepare for Q&A"""
    global conversation_chain, current_mode, current_model

    data = request.json
    url = data.get('url')
    mode = data.get('mode', 'news')  # 'news' or 'youtube'
    auto_send_telegram = data.get('auto_send_telegram', False)  # Only true for auto-processor

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    if mode not in ['news', 'youtube']:
        return jsonify({'error': 'Invalid mode. Use "news" or "youtube"'}), 400

    try:
        # Initialize conversation chain for the selected mode
        conversation_chain = create_runnable_chain(mode)
        current_mode = mode

        # Step 1: Fetch raw content
        if mode == 'news':
            print(f"[INFO] Loading news article from: {url}")
            documents = read_website_content(url)
            if not documents:
                print(f"[ERROR] Failed to load article from: {url}")
                return jsonify({'error': 'Could not load article from URL'}), 400

            raw_content = documents[0].page_content
            print(f"[SUCCESS] Article loaded successfully. Length: {len(raw_content)} characters")

        elif mode == 'youtube':
            print(f"[INFO] Fetching YouTube transcript from: {url}")
            raw_content = get_youtube_transcript(url)

            # Check if transcript fetch failed
            if raw_content.startswith("No transcript available") or raw_content.startswith("Error fetching") or raw_content.startswith("Invalid YouTube"):
                print(f"[ERROR] YouTube transcript fetch failed: {raw_content}")
                return jsonify({
                    'error': raw_content,
                    'success': False
                }), 400
            
            print(f"[SUCCESS] YouTube transcript fetched successfully. Length: {len(raw_content)} characters")

        # Step 2: Condense content using condenser_service (no model conversation yet)
        print(f"[INFO] Condensing content using condenser service...")
        condensed_content = condense_content(raw_content, current_model)
        print(f"[SUCCESS] Content condensed. Original: {len(raw_content)} chars -> Condensed: {len(condensed_content)} chars")

        # Calculate word counts
        raw_word_count = len(raw_content.split())
        condensed_word_count = len(condensed_content.split())
        
        # Step 3: Generate audio for condensed content
        audio_file = None
        audio_file_path = None
        print(f"[INFO] Generating audio for condensed content ({len(condensed_content)} chars)...")
        audio_start_time = time.time()
        
        try:
            audio = generate_audio(condensed_content)
            audio_file_path = create_audio_file(audio)
            audio_file = os.path.basename(audio_file_path)
            
            audio_time = time.time() - audio_start_time
            print(f"[SUCCESS] Audio generated: {audio_file}")
            print(f"[TIME] Audio generation took: {audio_time:.2f}s")
        except Exception as e:
            error_msg = f"Audio generation failed: {e}"
            print(f"[ERROR] {error_msg}")
            if auto_send_telegram:
                return jsonify({'error': error_msg, 'success': False}), 422
            # For manual load, audio failure is not critical
            print(f"[WARNING] Continuing without audio")
        
        # Step 4: Send to Telegram only if auto_send_telegram is True
        if auto_send_telegram:
            print(f"[INFO] Auto-sending to Telegram...")
            try:
                chat_id = os.getenv('TELEGRAM_CHAT_ID')
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                channel_id = os.getenv('TELEGRAM_CHANNEL_ID')  # Optional channel for posting
                
                if not chat_id or not bot_token:
                    error_msg = "Telegram credentials not configured (TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN required)"
                    print(f"[ERROR] {error_msg}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                if not audio_file_path:
                    error_msg = "Cannot send to Telegram: Audio file not generated"
                    print(f"[ERROR] {error_msg}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                content_type = 'Article' if mode == 'news' else 'YouTube Video'
                message = f"📝 Condensed {content_type}\n\n{condensed_content}"
                
                telegram_success = send_telegram_with_audio(
                    chat_id=chat_id,
                    message=message,
                    audio_file_path=audio_file_path,
                    bot_token=bot_token,
                    source_url=url,
                    channel_id=channel_id if channel_id else None
                )
                
                if not telegram_success:
                    error_msg = "Failed to send content to Telegram"
                    print(f"[ERROR] {error_msg}")
                    # Create backup file
                    try:
                        backup_path = create_backup_file(url, condensed_content, audio_file_path)
                        error_msg = f"Failed to send to Telegram. Backup created at: {backup_path}"
                        print(f"[BACKUP] {error_msg}")
                    except Exception as backup_error:
                        print(f"[ERROR] Failed to create backup: {backup_error}")
                    return jsonify({'error': error_msg, 'success': False}), 422
                
                print(f"[SUCCESS] Content sent to Telegram successfully")
            except Exception as e:
                error_msg = f"Telegram sending failed: {e}"
                print(f"[ERROR] {error_msg}")
                # Create backup file
                try:
                    backup_path = create_backup_file(url, condensed_content, audio_file_path)
                    error_msg = f"Telegram error: {e}. Backup created at: {backup_path}"
                    print(f"[BACKUP] {error_msg}")
                except Exception as backup_error:
                    print(f"[ERROR] Failed to create backup: {backup_error}")
                return jsonify({'error': error_msg, 'success': False}), 422
        
        # Step 5: Store condensed content in conversation memory for future Q&A
        # The memory now contains: system prompt (from create_runnable_chain) + condensed input
        session_history.add_user_message(f"Here is the condensed {'article' if mode == 'news' else 'video transcript'} content (Original: {raw_word_count} words, Condensed: {condensed_word_count} words):\n\n{condensed_content}")
        session_history.add_ai_message(f"I have received and processed the condensed content. I'm ready to answer your questions about it.")
        print(f"[INFO] Condensed content added to memory. Ready for Q&A.")

        # Return success only if everything succeeded
        return jsonify({
            'content': condensed_content,
            'mode': mode,
            'word_count': condensed_word_count,
            'original_word_count': raw_word_count,
            'audio_file': audio_file,
            'audio_time': round(audio_time, 2) if audio_time else None,
            'success': True
        })

    except ValueError as e:
        # ValueError is raised when thinking tokens aren't removed properly
        print(f"[ERROR] Thinking token validation failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 422
    except Exception as e:
        print(f"[ERROR] load_content failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/send_email', methods=['POST'])
def send_email():
    """Send condensed content with audio via email"""
    data = request.json
    audio_file = data.get('audio_file')
    content = data.get('content')
    mode = data.get('mode', 'news')
    url = data.get('url', '')
    
    # Get recipient email from environment variable
    recipient_email = os.getenv('RECIPIENT_GMAIL_ADDRESS')
    
    if not recipient_email:
        print("[ERROR] RECIPIENT_GMAIL_ADDRESS environment variable not set")
        return jsonify({'error': 'Recipient email not configured. Set RECIPIENT_GMAIL_ADDRESS environment variable.', 'success': False}), 500
    
    if not audio_file:
        return jsonify({'error': 'Audio file not found', 'success': False}), 400
    
    try:
        # Build audio file path
        audio_file_path = os.path.join('kokoro_outputs', audio_file)
        
        if not os.path.exists(audio_file_path):
            return jsonify({'error': 'Audio file does not exist', 'success': False}), 404
        
        # Create email subject and body
        content_type = 'Article' if mode == 'news' else 'Video Transcript'
        subject = f'Condensed {content_type} with Audio'
        
        # Truncate content if too long for email body
        body_content = content[:2000] + '...' if len(content) > 2000 else content
        
        # Include source URL in email body
        url_section = f"\n\nSource URL: {url}\n" if url else ""
        body_text = f"Here is the condensed {content_type.lower()}:{url_section}\n\n{body_content}\n\nPlease find the audio file attached."
        
        print(f"[INFO] Sending email to {recipient_email}...")
        success = send_email_with_audio(
            recipient_email=recipient_email,
            subject=subject,
            body_text=body_text,
            audio_file_path=audio_file_path
        )
        
        if success:
            print(f"[SUCCESS] Email sent to {recipient_email}")
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            print(f"[ERROR] Failed to send email to {recipient_email}")
            return jsonify({'error': 'Failed to send email. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_email endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/send_telegram', methods=['POST'])
def send_telegram():
    """Send condensed content with audio via Telegram"""
    data = request.json
    audio_file = data.get('audio_file')
    content = data.get('content')
    mode = data.get('mode', 'news')
    url = data.get('url', '')
    
    # Get Telegram credentials from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')  # Optional channel for posting
    
    if not bot_token:
        print("[ERROR] TELEGRAM_BOT_TOKEN environment variable not set")
        return jsonify({'error': 'Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.', 'success': False}), 500
    
    if not chat_id:
        print("[ERROR] TELEGRAM_CHAT_ID environment variable not set")
        return jsonify({'error': 'Telegram chat ID not configured. Set TELEGRAM_CHAT_ID environment variable.', 'success': False}), 500
    
    if not audio_file:
        return jsonify({'error': 'Audio file not found', 'success': False}), 400
    
    try:
        # Build audio file path
        audio_file_path = os.path.join('kokoro_outputs', audio_file)
        
        if not os.path.exists(audio_file_path):
            return jsonify({'error': 'Audio file does not exist', 'success': False}), 404
        
        # Create message text
        content_type = 'Article' if mode == 'news' else 'Video Transcript'
        
        # Don't include URL in message since it will be sent separately
        message = f"📝 Condensed {content_type}\n\n{content}"
        
        print(f"[INFO] Sending to Telegram chat {chat_id}...")
        print(f"[DEBUG] Message length: {len(message)} characters")
        success = send_telegram_with_audio(
            chat_id=chat_id,
            message=message,
            audio_file_path=audio_file_path,
            bot_token=bot_token,
            source_url=url,
            channel_id=channel_id if channel_id else None
        )
        
        if success:
            print(f"[SUCCESS] Telegram message sent to {chat_id}")
            return jsonify({'success': True, 'message': 'Telegram message sent successfully'})
        else:
            print(f"[ERROR] Failed to send Telegram message to {chat_id}")
            return jsonify({'error': 'Failed to send Telegram message. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_telegram endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/send_quick_email', methods=['POST'])
def send_quick_email():
    """Send quick email with custom message and attachments from UI"""
    message = request.form.get('message')
    files = request.files.getlist('attachments')
    
    # Get recipient email from environment variable
    recipient_email = os.getenv('RECIPIENT_GMAIL_ADDRESS')
    
    if not recipient_email:
        print("[ERROR] RECIPIENT_GMAIL_ADDRESS environment variable not set")
        return jsonify({'error': 'Recipient email not configured. Set RECIPIENT_GMAIL_ADDRESS environment variable.', 'success': False}), 500
    
    if not message:
        return jsonify({'error': 'Message is required', 'success': False}), 400
    
    try:
        # Save uploaded files temporarily
        temp_dir = 'temp_attachments'
        os.makedirs(temp_dir, exist_ok=True)
        
        attachment_paths = []
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                attachment_paths.append(file_path)
                print(f"[INFO] Saved temporary attachment: {file.filename}")
        
        # Send email
        subject = 'Quick Message from Content Analyzer'
        
        print(f"[INFO] Sending quick email to {recipient_email}...")
        success = send_email_with_attachments(
            recipient_email=recipient_email,
            subject=subject,
            body_text=message,
            attachment_paths=attachment_paths if attachment_paths else None
        )
        
        # Clean up temporary files
        for file_path in attachment_paths:
            try:
                os.remove(file_path)
                print(f"[INFO] Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file {file_path}: {e}")
        
        # Remove temp directory if empty
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        if success:
            print(f"[SUCCESS] Quick email sent to {recipient_email}")
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            print(f"[ERROR] Failed to send quick email to {recipient_email}")
            return jsonify({'error': 'Failed to send email. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_quick_email endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/send_quick_telegram', methods=['POST'])
def send_quick_telegram():
    """Send quick Telegram message with custom text and attachments from UI"""
    message = request.form.get('message')
    files = request.files.getlist('attachments')
    
    # Get Telegram credentials from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token:
        print("[ERROR] TELEGRAM_BOT_TOKEN environment variable not set")
        return jsonify({'error': 'Telegram bot token not configured. Set TELEGRAM_BOT_TOKEN environment variable.', 'success': False}), 500
    
    if not chat_id:
        print("[ERROR] TELEGRAM_CHAT_ID environment variable not set")
        return jsonify({'error': 'Telegram chat ID not configured. Set TELEGRAM_CHAT_ID environment variable.', 'success': False}), 500
    
    if not message:
        return jsonify({'error': 'Message is required', 'success': False}), 400
    
    try:
        # Save uploaded files temporarily
        temp_dir = 'temp_attachments'
        os.makedirs(temp_dir, exist_ok=True)
        
        attachment_paths = []
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                attachment_paths.append(file_path)
                print(f"[INFO] Saved temporary attachment: {file.filename}")
        
        # Send Telegram message
        print(f"[INFO] Sending quick Telegram message to {chat_id}...")
        success = send_telegram_with_attachments(
            chat_id=chat_id,
            message=message,
            attachment_paths=attachment_paths if attachment_paths else None,
            bot_token=bot_token
        )
        
        # Clean up temporary files
        for file_path in attachment_paths:
            try:
                os.remove(file_path)
                print(f"[INFO] Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file {file_path}: {e}")
        
        # Remove temp directory if empty
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        if success:
            print(f"[SUCCESS] Quick Telegram message sent to {chat_id}")
            return jsonify({'success': True, 'message': 'Telegram message sent successfully'})
        else:
            print(f"[ERROR] Failed to send quick Telegram message to {chat_id}")
            return jsonify({'error': 'Failed to send Telegram message. Check server logs for details.', 'success': False}), 500
            
    except Exception as e:
        print(f"[ERROR] send_quick_telegram endpoint failed: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


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
        response_text, thinking_tokens_removed = remove_thinking_tokens(response_text)
        if not thinking_tokens_removed:
            error_msg = "Failed to remove thinking tokens from LLM response"
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg, 'success': False}), 422
        
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
    print("[INFO] streamChat endpoint called")
    # Check if LLM server is accessible first
    if not check_llm_server():
        print("[ERROR] LLM server not accessible")
        return jsonify({
            'error': 'Local LLM server is not running. Please start LM Studio and load a model at http://localhost:1234'
        }), 503

    data = request.json
    user_input = data.get('message')
    generate_audio_flag = data.get('generate_audio', False)
    print(f"[INFO] User input: '{user_input[:50]}...' | Audio: {generate_audio_flag}")

    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    def stream_response():
        try:
            print("[DEBUG] stream_response: Starting generator")
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
            print(f"[INFO] Streaming complete. Time: {llm_time:.2f}s, Chunks: {total_chunk_size} chars")

            # Remove thinking tokens from complete response
            complete_response_text, thinking_tokens_removed = remove_thinking_tokens(complete_response_text)
            if not thinking_tokens_removed:
                error_msg = "Failed to remove thinking tokens from LLM response"
                print(f"[ERROR] {error_msg}")
                error_data = {'error': error_msg}
                yield f"data:{json.dumps(error_data)}\n\n"
                return
            print(f"[DEBUG] After thinking token removal: {len(complete_response_text)} chars")

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
    print(f"[INFO] Audio file requested: {filename}")
    audio_dir = 'kokoro_outputs'
    file_path = os.path.join(audio_dir, filename)

    if os.path.exists(file_path):
        print(f"[DEBUG] Serving audio file: {file_path}")
        return send_file(file_path, mimetype='audio/wav')

    print(f"[ERROR] Audio file not found: {file_path}")
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation memory and reset mode"""
    print("[INFO] Clearing conversation memory")
    global conversation_chain, current_mode
    
    if session_history:
        session_history.clear()
        print("[DEBUG] Session history cleared")
    
    conversation_chain = None
    current_mode = None
    print("[INFO] Conversation state reset")

    return jsonify({'success': True, 'message': 'Conversation cleared'})

@app.route('/text_to_audio', methods=['POST'])
def text_to_audio():
    """Convert text to audio file"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        print(f"[INFO] text_to_audio: Converting {len(text)} chars to audio")
        print(f"[DEBUG] Text preview: {text[:100]}...")
        
        # For very large text, split into chunks
        MAX_CHARS = 10000
        
        if len(text) > MAX_CHARS:
            print(f"[WARNING] Text exceeds {MAX_CHARS} chars, splitting into chunks...")
            sentences = text.replace('\n', ' ').split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < MAX_CHARS:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"[INFO] Split into {len(chunks)} chunks")
            
            import numpy as np
            audio_segments = []
            
            for idx, chunk in enumerate(chunks, 1):
                print(f"[INFO] Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
                chunk_audio = generate_audio(chunk)
                print(f"[DEBUG] Chunk {idx} audio shape: {chunk_audio.shape}")
                audio_segments.append(chunk_audio)
            
            audio = np.concatenate(audio_segments)
            print(f"[SUCCESS] All chunks processed and concatenated: {audio.shape}")
        else:
            print(f"[INFO] Generating audio directly (text < {MAX_CHARS} chars)")
            audio = generate_audio(text)
            import numpy as np
            print(f"[DEBUG] Generated audio shape: {audio.shape}, sum: {np.sum(np.abs(audio))}")
        
        if audio is None or (hasattr(audio, 'size') and audio.size == 0):
            raise ValueError("Generated audio is empty")
        
        audio_file_path = create_audio_file(audio)
        audio_filename = os.path.basename(audio_file_path)
        
        print(f"[SUCCESS] Audio file created: {audio_filename}")
        
        return jsonify({
            'success': True,
            'audio_file': audio_filename,
            'message': f'Audio generated successfully ({len(text)} chars)'
        })
        
    except Exception as e:
        error_msg = f"Text-to-audio conversion failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)