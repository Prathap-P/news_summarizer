import re
import os
from datetime import datetime
from pathlib import Path


def remove_thinking_tokens(text: str) -> tuple[str, bool]:
    """
    Extract final script from LLM response by finding content within <final_script> tags.
    
    The LLM returns the final cleaned output wrapped in <final_script></final_script> tags.
    This function extracts only the content between the LAST opening and LAST closing tag,
    discarding all reasoning, thinking tokens, and other artifacts.

    Args:
        text: Raw LLM response text containing <final_script> tags

    Returns:
        Tuple of (cleaned_text, success_bool) where success_bool indicates if tags were found
    """
    if not text:
        print("[DEBUG] remove_thinking_tokens: Empty text received")
        return text, False

    print(f"[DEBUG] remove_thinking_tokens: Processing {len(text)} characters")
    original_length = len(text)

    # Find the last occurrence of opening and closing tags
    opening_tag = '<final_script>'
    closing_tag = '</final_script>'
    
    last_open_index = text.lower().rfind(opening_tag.lower())
    last_close_index = text.lower().rfind(closing_tag.lower())
    
    if last_open_index != -1 and last_close_index != -1 and last_close_index > last_open_index:
        # Extract content between last opening tag and last closing tag
        start_pos = last_open_index + len(opening_tag)
        final_content = text[start_pos:last_close_index].strip()
        
        print(f"[INFO] Found <final_script> tags, extracting content from last occurrence")
        print(f"[CLEANUP] Extracted {len(final_content)} chars from final_script tag (removed {original_length - len(final_content)} chars)")
        return final_content, True
    else:
        print("[WARNING] No valid <final_script> tags found, thinking tokens not properly removed")
        print(f"[DEBUG] Original text content:\n{text}")
        return text.strip(), False


def create_backup_file(url: str, content: str, audio_file_path: str) -> str:
    """
    Create a backup file for content and audio when Telegram sending fails.
    
    Args:
        url: The source URL
        content: The condensed content text
        audio_file_path: Path to the audio file
        
    Returns:
        Path to the created backup file
    """
    # Create backup directory if it doesn't exist
    backup_dir = Path("backup_content")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate a safe filename from the URL
    # Remove protocol and special characters
    safe_name = re.sub(r'https?://', '', url)
    safe_name = re.sub(r'[^\w\-_]', '_', safe_name)
    
    # Limit filename length
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    
    # Add timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.txt"
    
    backup_path = backup_dir / filename
    
    # Write content and audio path to the backup file
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(f"Source URL: {url}\n")
        f.write(f"Backup Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(content)
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Audio File Path: {audio_file_path}\n")
    
    print(f"[BACKUP] Created backup file: {backup_path}")
    return str(backup_path)
