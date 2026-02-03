from langchain_text_splitters import RecursiveCharacterTextSplitter

from system_prompts import *
from utils import remove_thinking_tokens

# Configuration
REDUCE_BATCH_SIZE = 5  # Number of chunks to reduce per batch

def condense_content(content: str, current_model):
    print(f"[INFO] condense_content: Starting condensation for {len(content)} chars")
    chunks = split_content(content)
    print(f"[INFO] Content split into {len(chunks)} chunks")
    processed_chunks = []
    map_prompt = map_reduce_custom_prompts["map_prompt"]
    reduce_prompt = map_reduce_custom_prompts["reduce_prompt"]

    #Map phase
    print("[INFO] Starting MAP phase")
    for idx, chunk in enumerate(chunks, 1):
        print(f"[DEBUG] Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        map_input = f"""
            System:
            {yt_transcript_shortener_system_message}
            Input:
            {map_prompt.replace('{chunk_text}', chunk)}
        """

        # Stream the response and accumulate all chunks
        print(f"[DEBUG] Streaming MAP chunk {idx}...")
        chunk_response_text = ""
        for streamed_chunk in current_model.stream(map_input):
            # Extract content from AIMessageChunk
            if hasattr(streamed_chunk, 'content'):
                chunk_response_text += streamed_chunk.content
            else:
                chunk_response_text += str(streamed_chunk)
        
        print(f"[DEBUG] MAP chunk {idx} streaming complete: {len(chunk_response_text)} chars")
        cleaned_chunk_response, success = remove_thinking_tokens(chunk_response_text)
        if not success:
            error_msg = f"Failed to remove thinking tokens from MAP chunk {idx}/{len(chunks)}"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        processed_chunks.append(cleaned_chunk_response)
        print(f"[DEBUG] Cleaned chunk {idx} processed: {len(cleaned_chunk_response)} chars")

    print(f"[INFO] MAP phase complete. {len(processed_chunks)} chunks processed")
    #Reduce phase
    print("[INFO] Starting REDUCE phase")
    
    reduce_prompt = map_reduce_custom_prompts["reduce_prompt"]
    reduce_with_context_prompt = map_reduce_custom_prompts["reduce_with_context_prompt"]
    
    # Check if we need batch processing
    if len(processed_chunks) <= REDUCE_BATCH_SIZE:
        # Single batch - use original logic
        print(f"[INFO] Single batch mode ({len(processed_chunks)} chunks)")
        combined_chunks = "\n\n---\n\n".join(processed_chunks)
        print(f"[DEBUG] Combined chunks: {len(combined_chunks)} chars")
        
        reduce_input = f"""
                System:
                {yt_transcript_shortener_system_message}
                Input:
                {reduce_prompt.replace('{combined_map_results}', combined_chunks)}
            """
        
        print(f"[DEBUG] Streaming REDUCE phase...")
        reduce_response_text = ""
        for streamed_chunk in current_model.stream(reduce_input):
            if hasattr(streamed_chunk, 'content'):
                reduce_response_text += streamed_chunk.content
            else:
                reduce_response_text += str(streamed_chunk)
        
        print(f"[DEBUG] REDUCE streaming complete: {len(reduce_response_text)} chars")
        cleaned_reduce_response, success = remove_thinking_tokens(reduce_response_text)
        if not success:
            error_msg = "Failed to remove thinking tokens from REDUCE phase"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        final_output = cleaned_reduce_response
    else:
        # Batch mode with context continuity
        num_batches = (len(processed_chunks) + REDUCE_BATCH_SIZE - 1) // REDUCE_BATCH_SIZE
        print(f"[INFO] Batch mode: Processing {len(processed_chunks)} chunks in {num_batches} batches (with context continuity)")
        
        batch_results = []
        previous_context = ""
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * REDUCE_BATCH_SIZE
            end_idx = min((batch_idx + 1) * REDUCE_BATCH_SIZE, len(processed_chunks))
            batch_chunks = processed_chunks[start_idx:end_idx]
            
            print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches} ({len(batch_chunks)} chunks)...")
            
            combined_batch = "\n\n---\n\n".join(batch_chunks)
            print(f"[DEBUG] Batch {batch_idx + 1} size: {len(combined_batch)} chars")
            
            # First batch uses standard reduce, subsequent batches use context-aware reduce
            if batch_idx == 0:
                prompt_to_use = reduce_prompt.replace('{combined_map_results}', combined_batch)
            else:
                # Limit previous context to last 3000 chars to avoid context overflow
                context_snippet = previous_context[-3000:] if len(previous_context) > 3000 else previous_context
                prompt_to_use = reduce_with_context_prompt.replace('{previous_context}', context_snippet)
                prompt_to_use = prompt_to_use.replace('{combined_map_results}', combined_batch)
            
            reduce_input = f"""
                    System:
                    {yt_transcript_shortener_system_message}
                    Input:
                    {prompt_to_use}
                """
            
            print(f"[DEBUG] Streaming batch {batch_idx + 1}...")
            batch_response_text = ""
            for streamed_chunk in current_model.stream(reduce_input):
                if hasattr(streamed_chunk, 'content'):
                    batch_response_text += streamed_chunk.content
                else:
                    batch_response_text += str(streamed_chunk)
            
            print(f"[DEBUG] Batch {batch_idx + 1} streaming complete: {len(batch_response_text)} chars")
            cleaned_batch_response, success = remove_thinking_tokens(batch_response_text)
            if not success:
                error_msg = f"Failed to remove thinking tokens from batch {batch_idx + 1}/{num_batches}"
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
            
            batch_results.append(cleaned_batch_response)
            previous_context = cleaned_batch_response  # Use this batch's output as context for next
            print(f"[SUCCESS] Batch {batch_idx + 1} reduced: {len(cleaned_batch_response)} chars")
        
        # Concatenate all batch results
        final_output = "\n\n".join(batch_results)
        print(f"[INFO] All batches combined: {len(final_output)} chars")

    print(f"[INFO] REDUCE phase complete: {len(final_output)} chars")
    print(f"[SUCCESS] Condensation complete. Original: {len(content)} -> Final: {len(final_output)} chars")
    return final_output

def split_content(content: str):
    chunk_size = 10000
    chunk_overlap = 200
    print(f"[DEBUG] split_content: Splitting {len(content)} chars with chunk_size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_text(content)
    print(f"[DEBUG] split_content: Created {len(chunks)} chunks")
    return chunks