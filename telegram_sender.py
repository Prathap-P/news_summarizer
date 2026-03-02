import os
import requests
import time
from typing import Optional, List
from pathlib import Path

# Import compress_audio from utils
try:
    from utils import compress_audio
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    print("[WARNING] compress_audio not available - large files will fail")

# Telegram limits
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_MAX_FILE_SIZE_MB = 50


def get_discussion_group_id(channel_id: str, bot_token: str) -> Optional[str]:
    """
    Get the linked discussion group ID for a Telegram channel.
    
    Args:
        channel_id: Channel ID (e.g., '-1003892267199')
        bot_token: Telegram bot token
        
    Returns:
        Discussion group chat ID if linked, None otherwise
    """
    try:
        get_chat_url = f"https://api.telegram.org/bot{bot_token}/getChat"
        response = requests.post(get_chat_url, data={'chat_id': channel_id})
        
        if response.ok:
            chat_info = response.json()
            if 'result' in chat_info and 'linked_chat_id' in chat_info['result']:
                linked_id = chat_info['result']['linked_chat_id']
                print(f"[INFO] Found linked discussion group: {linked_id}")
                return str(linked_id)
            else:
                print(f"[WARNING] Channel {channel_id} has no linked discussion group")
                return None
        else:
            print(f"[ERROR] Failed to get chat info: {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception getting discussion group: {str(e)}")
        return None


def split_message(message: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into multiple chunks respecting Telegram's character limit.
    Maximizes use of character limit and splits at word boundaries.
    
    Args:
        message: The message to split
        max_length: Maximum length per message chunk (default: 4096)
        
    Returns:
        List of message chunks
    """
    if len(message) <= max_length:
        return [message]
    
    chunks = []
    remaining = message
    
    while remaining:
        # If remaining text fits in one message, add it and break
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Take as much as possible up to max_length
        chunk = remaining[:max_length]
        
        # Find the last space to avoid breaking words
        last_space = chunk.rfind(' ')
        last_newline = chunk.rfind('\n')
        
        # Use the last whitespace (prefer newline over space)
        split_pos = max(last_space, last_newline)
        
        if split_pos > 0:
            # Split at word boundary
            chunks.append(chunk[:split_pos])
            remaining = remaining[split_pos:].lstrip()
        else:
            # No space found, force split at max_length
            chunks.append(chunk)
            remaining = remaining[max_length:]
    
    return chunks


def send_telegram_with_audio(
    chat_id: str,
    message: str,
    audio_file_path: str,
    bot_token: Optional[str] = None,
    source_url: Optional[str] = None,
    channel_id: Optional[str] = None
) -> bool:
    """
    Send a Telegram message with a single audio file attachment.
    
    If channel_id provided:
        1) Send URL link to channel (broadcast post)
        2) Send audio file to linked discussion group (as comment)
        3) Send text content to discussion group (as comment)
    
    If no channel_id:
        1) Send URL link to chat_id
        2) Send audio file to chat_id
        3) Send text content to chat_id
    
    Args:
        chat_id: Telegram chat ID (fallback if no channel_id)
        message: Text message to send
        audio_file_path: Path to the audio file to attach
        bot_token: Telegram bot token (defaults to env var TELEGRAM_BOT_TOKEN)
        source_url: Optional source URL to send first
        channel_id: Optional channel ID for posting to channel + discussion group
        
    Returns:
        True if message sent successfully, False otherwise
        
    Environment Variables:
        TELEGRAM_BOT_TOKEN: Your Telegram bot token from @BotFather
    """
    # Get bot token from env variable if not provided
    bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Validate inputs
    if not bot_token:
        print("[ERROR] Bot token not provided. Set TELEGRAM_BOT_TOKEN env variable or pass bot_token parameter")
        return False
    
    if not chat_id:
        print("[ERROR] Chat ID is required")
        return False
    
    # Validate audio file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return False
    
    # Check file size and compress if needed
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Audio file size: {file_size_mb:.2f} MB")
    
    compressed_file = None
    if file_size_mb > TELEGRAM_MAX_FILE_SIZE_MB:
        print(f"[WARNING] File exceeds {TELEGRAM_MAX_FILE_SIZE_MB}MB Telegram limit")
        
        if not COMPRESSION_AVAILABLE:
            print(f"[ERROR] Compression not available. Cannot send file >50MB")
            return False
        
        print(f"[INFO] Attempting to compress audio...")
        compressed_file = compress_audio(str(audio_path), bitrate="64k")
        
        if not compressed_file:
            print(f"[ERROR] Compression failed. Cannot send file.")
            return False
        
        # Check if compressed file is now small enough
        compressed_size_mb = Path(compressed_file).stat().st_size / (1024 * 1024)
        if compressed_size_mb > TELEGRAM_MAX_FILE_SIZE_MB:
            print(f"[WARNING] Compressed file still too large ({compressed_size_mb:.2f}MB)")
            print(f"[INFO] Trying higher compression (32k bitrate)...")
            
            # Try with lower bitrate
            os.remove(compressed_file)  # Remove previous attempt
            compressed_file = compress_audio(str(audio_path), bitrate="32k")
            
            if compressed_file:
                compressed_size_mb = Path(compressed_file).stat().st_size / (1024 * 1024)
                if compressed_size_mb > TELEGRAM_MAX_FILE_SIZE_MB:
                    print(f"[ERROR] Even with 32k bitrate, file is {compressed_size_mb:.2f}MB (still >50MB)")
                    os.remove(compressed_file)
                    return False
            else:
                return False
        
        # Use compressed file
        audio_path = Path(compressed_file)
        audio_file_path = compressed_file
        print(f"[SUCCESS] Using compressed audio: {compressed_size_mb:.2f} MB")
    
    try:
        text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        audio_url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
        
        # Determine target chat for comments (discussion group or fallback to chat_id)
        comment_chat_id = chat_id
        
        # If channel_id provided, use channel + discussion group workflow
        if channel_id:
            print(f"[INFO] Using channel mode: post to channel {channel_id}, reply in group {chat_id}")
            
            # Step 1: Send URL link to CHANNEL (will auto-forward to discussion group)
            if not source_url:
                print(f"[ERROR] source_url is required for channel mode")
                return False
                
            print(f"[INFO] Sending source URL to channel {channel_id}...")
            url_response = requests.post(text_url, data={
                'chat_id': channel_id,
                'text': f"🔗 Source: {source_url}"
            })
            
            if not url_response.ok:
                print(f"[ERROR] Failed to send URL to channel: {url_response.text}")
                return False
            
            # Get channel message_id from response
            channel_message_id = url_response.json()['result']['message_id']
            print(f"[SUCCESS] Source URL posted to channel, message_id: {channel_message_id}")
            
            # Step 2: Wait for auto-forward to discussion group
            print(f"[INFO] Waiting 3 seconds for auto-forward to discussion group...")
            time.sleep(3)
            
            # Step 3: Find the forwarded message in discussion group using getUpdates
            print(f"[INFO] Looking for forwarded message in discussion group {chat_id}...")
            get_updates_url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            
            discussion_message_id = None
            max_attempts = 3
            
            for attempt in range(1, max_attempts + 1):
                print(f"[INFO] Attempt {attempt}/{max_attempts}...")
                
                updates_response = requests.get(get_updates_url, params={'limit': 100}, timeout=10)
                
                if updates_response.ok:
                    updates = updates_response.json().get('result', [])
                    
                    # Find message with forward_from_message_id matching our channel post
                    for update in reversed(updates):
                        if 'message' in update:
                            msg = update['message']
                            msg_chat_id = str(msg.get('chat', {}).get('id', ''))
                            forward_from_message_id = msg.get('forward_from_message_id')
                            
                            # Check if this is the forwarded message from our channel post
                            if (msg_chat_id == str(chat_id) and 
                                forward_from_message_id == channel_message_id):
                                
                                discussion_message_id = msg['message_id']
                                print(f"[SUCCESS] Found forwarded message: channel_msg={channel_message_id} → discussion_msg={discussion_message_id}")
                                break
                
                if discussion_message_id:
                    break
                
                if attempt < max_attempts:
                    print(f"[INFO] Not found yet, waiting 2 more seconds...")
                    time.sleep(2)
            
            if not discussion_message_id:
                print(f"[WARNING] Could not find forwarded message after {max_attempts} attempts")
                print(f"[INFO] Sending audio and text without threading")
            
            # Step 4: Send audio file to discussion group as reply
            print(f"[INFO] Sending audio to discussion group {chat_id}...")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': (audio_path.name, audio_file, 'audio/wav')}
                data = {'chat_id': chat_id}
                
                if discussion_message_id:
                    data['reply_to_message_id'] = discussion_message_id
                
                audio_response = requests.post(audio_url, data=data, files=files)
                
                if not audio_response.ok:
                    print(f"[ERROR] Failed to send audio: {audio_response.text}")
                    return False
            
            print(f"[SUCCESS] Audio sent to discussion group")
            
            # Step 5: Send text content to discussion group as reply
            message_chunks = split_message(message)
            
            print(f"[DEBUG] Message length: {len(message)} chars")
            print(f"[DEBUG] Split into {len(message_chunks)} chunks")
            for idx, chunk in enumerate(message_chunks, 1):
                print(f"[DEBUG] Chunk {idx} length: {len(chunk)} chars")
            
            print(f"[INFO] Sending text content to discussion group ({len(message_chunks)} part(s))...")
            
            for i, chunk in enumerate(message_chunks, 1):
                print(f"[INFO] Sending message part {i}/{len(message_chunks)}...")
                
                data = {
                    'chat_id': chat_id,
                    'text': chunk
                }
                
                if discussion_message_id:
                    data['reply_to_message_id'] = discussion_message_id
                
                text_response = requests.post(text_url, data=data)
                
                if not text_response.ok:
                    print(f"[ERROR] Failed to send message part {i}: {text_response.text}")
                    return False
            
            print(f"[SUCCESS] All messages sent as replies ({len(message_chunks)} part(s))")
            
        else:
            # No channel mode - use old behavior
            print(f"[INFO] Using direct chat mode: sending to {chat_id}")
            
            # Step 1: Send URL link if provided
            if source_url:
                print(f"[INFO] Sending source URL to chat {chat_id}...")
                url_response = requests.post(text_url, data={
                    'chat_id': chat_id,
                    'text': f"🔗 Source: {source_url}"
                })
                
                if not url_response.ok:
                    print(f"[ERROR] Failed to send URL: {url_response.text}")
                    return False
                
                print(f"[SUCCESS] Source URL sent")
        
            # Step 2: Send audio file to chat_id
            print(f"[INFO] Sending audio file to {chat_id}: {audio_path.name}")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': (audio_path.name, audio_file, 'audio/wav')}
                data = {'chat_id': chat_id}
                
                audio_response = requests.post(audio_url, data=data, files=files)
                
                if not audio_response.ok:
                    print(f"[ERROR] Failed to send audio: {audio_response.text}")
                    return False
            
            print(f"[SUCCESS] Audio file sent successfully")
            
            # Step 3: Send text content (split into chunks if needed)
            message_chunks = split_message(message)
            
            print(f"[DEBUG] Message length: {len(message)} chars")
            print(f"[DEBUG] Split into {len(message_chunks)} chunks")
            for idx, chunk in enumerate(message_chunks, 1):
                print(f"[DEBUG] Chunk {idx} length: {len(chunk)} chars")
            
            print(f"[INFO] Sending text content to {chat_id} ({len(message_chunks)} part(s))...")
            
            # Send all message chunks in order
            for i, chunk in enumerate(message_chunks, 1):
                print(f"[INFO] Sending message part {i}/{len(message_chunks)}...")
                text_response = requests.post(text_url, data={
                    'chat_id': chat_id,
                    'text': chunk
                })
                
                if not text_response.ok:
                    print(f"[ERROR] Failed to send message part {i}: {text_response.text}")
                    return False
            
            print(f"[SUCCESS] All message parts sent ({len(message_chunks)} message(s))")
        
        # Send separator messages to denote end of transaction
        # separator = "─" * 30
        # for _ in range(3):
        #     separator_response = requests.post(text_url, data={
        #         'chat_id': chat_id if not channel_id else channel_id,
        #         'text': separator
        #     })
        
        print(f"[SUCCESS] Message with audio sent successfully to Telegram")
        
        # Cleanup: Delete compressed file if it was created
        if compressed_file and os.path.exists(compressed_file):
            try:
                os.remove(compressed_file)
                print(f"[CLEANUP] Removed temporary compressed file: {Path(compressed_file).name}")
            except Exception as e:
                print(f"[WARNING] Could not remove compressed file: {e}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error occurred: {e}")
        # Cleanup compressed file on error
        if compressed_file and os.path.exists(compressed_file):
            try:
                os.remove(compressed_file)
            except:
                pass
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        # Cleanup compressed file on error
        if compressed_file and os.path.exists(compressed_file):
            try:
                os.remove(compressed_file)
            except:
                pass
        return False


def send_telegram_with_attachments(
    chat_id: str,
    message: str,
    attachment_paths: Optional[List[str]] = None,
    bot_token: Optional[str] = None
) -> bool:
    """
    Send a Telegram message with optional file attachments.
    
    Args:
        chat_id: Telegram chat ID or username (@username)
        message: Text message to send
        attachment_paths: Optional list of file paths to attach
        bot_token: Telegram bot token (defaults to env var TELEGRAM_BOT_TOKEN)
        
    Returns:
        True if message sent successfully, False otherwise
        
    Environment Variables:
        TELEGRAM_BOT_TOKEN: Your Telegram bot token from @BotFather
    """
    # Get bot token from env variable if not provided
    bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Validate inputs
    if not bot_token:
        print("[ERROR] Bot token not provided. Set TELEGRAM_BOT_TOKEN env variable or pass bot_token parameter")
        return False
    
    if not chat_id:
        print("[ERROR] Chat ID is required")
        return False
    
    try:
        text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Split message if needed
        message_chunks = split_message(message)
        
        print(f"[INFO] Sending message to Telegram chat {chat_id} ({len(message_chunks)} part(s))...")
        
        # Send all message chunks
        for i, chunk in enumerate(message_chunks, 1):
            print(f"[INFO] Sending message part {i}/{len(message_chunks)}...")
            text_response = requests.post(text_url, data={
                'chat_id': chat_id,
                'text': chunk
            })
            
            if not text_response.ok:
                print(f"[ERROR] Failed to send message part {i}: {text_response.text}")
                return False
        
        print(f"[SUCCESS] All message parts sent ({len(message_chunks)} message(s))")
        
        # Send attachments if provided
        if attachment_paths:
            for file_path in attachment_paths:
                if not os.path.exists(file_path):
                    print(f"[WARNING] Attachment not found, skipping: {file_path}")
                    continue
                
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                # Check file size (Telegram bot limit is 50MB)
                if file_size > 50 * 1024 * 1024:
                    print(f"[WARNING] File too large (>50MB), skipping: {file_name}")
                    continue
                
                print(f"[INFO] Sending file: {file_name} ({file_size / 1024:.2f} KB)")
                
                # Determine endpoint based on file type
                file_ext = file_path.lower()
                if file_ext.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
                    endpoint = 'sendAudio'
                    mime_type = 'audio/wav' if file_ext.endswith('.wav') else 'audio/mpeg'
                    file_key = 'audio'
                elif file_ext.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    endpoint = 'sendPhoto'
                    mime_type = 'image/jpeg'
                    file_key = 'photo'
                else:
                    endpoint = 'sendDocument'
                    mime_type = 'application/octet-stream'
                    file_key = 'document'
                
                url = f"https://api.telegram.org/bot{bot_token}/{endpoint}"
                
                with open(file_path, 'rb') as f:
                    files = {file_key: (file_name, f, mime_type)}
                    data = {'chat_id': chat_id}
                    
                    response = requests.post(url, data=data, files=files)
                    
                    if not response.ok:
                        print(f"[ERROR] Failed to send {file_name}: {response.text}")
                        continue
                
                print(f"[SUCCESS] Sent {file_name}")
        
        print(f"[SUCCESS] All messages and attachments sent successfully")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error occurred: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Telegram Sender with Attachments")
    print("=" * 60)
    
    # Check for environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("\n[SETUP] Required environment variables:")
        print("  export TELEGRAM_BOT_TOKEN='your-bot-token'")
        print("  export TELEGRAM_CHAT_ID='your-chat-id'")
        print("\n[NOTE] Get bot token from: @BotFather on Telegram")
        print("[NOTE] Get chat ID from: @userinfobot on Telegram")
        exit(1)
    
    # Example configuration
    message = input("\nEnter message to send: ").strip() or "Test message from Content Analyzer"
    audio_file = input("Enter path to audio file (or press Enter to skip): ").strip()
    
    if audio_file:
        # Send with audio
        success = send_telegram_with_audio(
            chat_id=chat_id,
            message=message,
            audio_file_path=audio_file
        )
    else:
        # Send text only
        success = send_telegram_with_attachments(
            chat_id=chat_id,
            message=message
        )
    
    if success:
        print("\n✓ Message sent successfully!")
    else:
        print("\n✗ Failed to send message")
