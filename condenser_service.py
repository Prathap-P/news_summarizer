from datetime import datetime
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from system_prompts import *
from utils import remove_thinking_tokens
from condensation_cache import save_checkpoint, MAX_RETRIES_PER_STEP

# Configuration
REDUCE_BATCH_SIZE = 3  # Number of chunks to reduce per batch (smaller = less hallucination)
FINAL_CONSOLIDATION_THRESHOLD = 150000  # Chars threshold to trigger final consolidation


def condense_content(
    content: str,
    current_model,
    checkpoint_key: Optional[str] = None,
    checkpoint: Optional[dict] = None,
) -> str:
    """Run map-reduce condensation, resuming from checkpoint if provided.

    Args:
        content:        Raw text to condense.
        current_model:  LangChain LLM instance.
        checkpoint_key: Key for atomic checkpoint saves.  None = no persistence.
        checkpoint:     Mutable checkpoint dict; updated in-place and saved after
                        every step so a crash loses at most one step's work.

    Returns:
        Condensed text string.
    """
    _has_checkpoint = checkpoint_key is not None and checkpoint is not None

    def _save() -> None:
        if _has_checkpoint:
            save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] condense_content: Starting condensation for {len(content)} chars")

    # ------------------------------------------------------------------
    # Stage 1 — split into chunks (idempotent; reuse stored split on resume)
    # ------------------------------------------------------------------
    if _has_checkpoint and checkpoint.get("map_chunks"):
        chunks = checkpoint["map_chunks"]
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: reusing {len(chunks)} stored chunks from checkpoint")
    else:
        chunks = split_content(content)
        if _has_checkpoint:
            checkpoint["map_chunks"] = chunks
            _save()
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Content split into {len(chunks)} chunks")

    map_prompt = map_reduce_custom_prompts["map_prompt"]
    reduce_prompt = map_reduce_custom_prompts["reduce_prompt"]
    reduce_with_context_prompt = map_reduce_custom_prompts["reduce_with_context_prompt"]

    # ------------------------------------------------------------------
    # Stage 2 — MAP phase
    # ------------------------------------------------------------------
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Starting MAP phase ({len(chunks)} chunks)")
    processed_chunks: list[str] = []

    for idx, chunk in enumerate(chunks):
        str_idx = str(idx)

        # Resume: skip already-completed chunks
        if _has_checkpoint and str_idx in checkpoint["map_results"]:
            cached = checkpoint["map_results"][str_idx]
            processed_chunks.append(cached)
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: MAP chunk {idx + 1}/{len(chunks)} already complete, skipping")
            continue

        # Retry cap: fail fast if this chunk has already blown its budget
        if _has_checkpoint:
            retries_used = checkpoint["map_retry_counts"].get(str_idx, 0)
            if retries_used >= MAX_RETRIES_PER_STEP:
                error_msg = (
                    f"MAP chunk {idx + 1}/{len(chunks)} exceeded max retries "
                    f"({MAX_RETRIES_PER_STEP}). Aborting — fix the model response or "
                    f"delete the checkpoint to start fresh."
                )
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

        print(f"[DEBUG] Processing MAP chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)")
        map_input = f"""
            System:
            {yt_transcript_shortener_system_message}
            Input:
            {map_prompt.replace('{chunk_text}', chunk)}
        """

        try:
            response = current_model.invoke(map_input)
            chunk_response_text = response.content
        except Exception as e:
            print(f"[ERROR] Model crashed during MAP chunk {idx + 1}/{len(chunks)}: {e}")
            if _has_checkpoint:
                checkpoint["map_retry_counts"][str_idx] = (
                    checkpoint["map_retry_counts"].get(str_idx, 0) + 1
                )
                _save()
            raise ValueError(f"Model crashed during MAP chunk {idx + 1}/{len(chunks)}: {e}")

        print(f"[DEBUG] MAP chunk {idx + 1} complete: {len(chunk_response_text)} chars")
        cleaned, success = remove_thinking_tokens(chunk_response_text)

        if not success:
            error_msg = f"Failed to remove thinking tokens from MAP chunk {idx + 1}/{len(chunks)}"
            print(f"[ERROR] {error_msg}")
            # Persist retry count before raising so the caller can resume
            if _has_checkpoint:
                checkpoint["map_retry_counts"][str_idx] = (
                    checkpoint["map_retry_counts"].get(str_idx, 0) + 1
                )
                _save()
            raise ValueError(error_msg)

        # Success — persist before moving on
        if _has_checkpoint:
            checkpoint["map_results"][str_idx] = cleaned
            _save()

        processed_chunks.append(cleaned)
        print(f"[DEBUG] MAP chunk {idx + 1} processed: {len(cleaned)} chars")

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] MAP phase complete. {len(processed_chunks)} chunks processed")

    # ------------------------------------------------------------------
    # Stage 3 — REDUCE phase
    # ------------------------------------------------------------------
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Starting REDUCE phase")

    if len(processed_chunks) <= REDUCE_BATCH_SIZE:
        # ---- Single-batch reduce ----
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Single batch mode ({len(processed_chunks)} chunks)")

        # Re-use cached result if it exists (covers single-batch reduce resume)
        if _has_checkpoint and checkpoint["reduce_results"].get("0"):
            final_output = checkpoint["reduce_results"]["0"]
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: single-batch REDUCE already complete, skipping")
        else:
            if _has_checkpoint:
                retry_count = checkpoint["reduce_retry_counts"].get("0", 0)
                if retry_count >= MAX_RETRIES_PER_STEP:
                    raise ValueError(
                        f"Single-batch REDUCE exceeded max retries ({MAX_RETRIES_PER_STEP})."
                    )

            combined_chunks = "\n\n---\n\n".join(processed_chunks)
            reduce_input = f"""
                    System:
                    {yt_transcript_shortener_system_message}
                    Input:
                    {reduce_prompt.replace('{combined_map_results}', combined_chunks)}
                """

            print(f"[DEBUG] Running REDUCE phase...")
            try:
                response = current_model.invoke(reduce_input)
                reduce_response_text = response.content
            except Exception as e:
                print(f"[ERROR] Model crashed during single-batch REDUCE: {e}")
                if _has_checkpoint:
                    checkpoint["reduce_retry_counts"]["0"] = (
                        checkpoint["reduce_retry_counts"].get("0", 0) + 1
                    )
                    _save()
                raise ValueError(f"Model crashed during single-batch REDUCE: {e}")

            print(f"[DEBUG] REDUCE complete: {len(reduce_response_text)} chars")
            cleaned_reduce, success = remove_thinking_tokens(reduce_response_text)
            if not success:
                error_msg = "Failed to remove thinking tokens from single-batch REDUCE phase"
                print(f"[ERROR] {error_msg}")
                if _has_checkpoint:
                    checkpoint["reduce_retry_counts"]["0"] = (
                        checkpoint["reduce_retry_counts"].get("0", 0) + 1
                    )
                    _save()
                raise ValueError(error_msg)

            if _has_checkpoint:
                checkpoint["reduce_results"]["0"] = cleaned_reduce
                checkpoint["reduce_batches_total"] = 1
                _save()

            final_output = cleaned_reduce

    else:
        # ---- Multi-batch reduce ----
        num_batches = (len(processed_chunks) + REDUCE_BATCH_SIZE - 1) // REDUCE_BATCH_SIZE
        print(
            f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Batch mode: {len(processed_chunks)} chunks "
            f"→ {num_batches} batches (with context continuity)"
        )

        if _has_checkpoint and checkpoint.get("reduce_batches_total") is None:
            checkpoint["reduce_batches_total"] = num_batches
            _save()

        # Seed previous_context from the last already-completed batch so that
        # a resumed run doesn't start batch N with empty context.
        completed_batch_indices = sorted(
            int(k) for k in checkpoint["reduce_results"]
        ) if _has_checkpoint else []
        previous_context = (
            checkpoint["reduce_results"][str(completed_batch_indices[-1])]
            if completed_batch_indices else ""
        )

        batch_results: list[str] = []

        for batch_idx in range(num_batches):
            str_batch = str(batch_idx)

            # Collect already-done batch result and keep previous_context in sync
            if _has_checkpoint and str_batch in checkpoint["reduce_results"]:
                cached_batch = checkpoint["reduce_results"][str_batch]
                batch_results.append(cached_batch)
                previous_context = cached_batch
                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Resuming: REDUCE batch {batch_idx + 1}/{num_batches} already complete, skipping"
                )
                continue

            # Retry cap
            if _has_checkpoint:
                retries_used = checkpoint["reduce_retry_counts"].get(str_batch, 0)
                if retries_used >= MAX_RETRIES_PER_STEP:
                    raise ValueError(
                        f"REDUCE batch {batch_idx + 1}/{num_batches} exceeded max retries "
                        f"({MAX_RETRIES_PER_STEP})."
                    )

            start_idx = batch_idx * REDUCE_BATCH_SIZE
            end_idx = min((batch_idx + 1) * REDUCE_BATCH_SIZE, len(processed_chunks))
            batch_chunks = processed_chunks[start_idx:end_idx]

            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"Processing REDUCE batch {batch_idx + 1}/{num_batches} ({len(batch_chunks)} chunks)..."
            )

            combined_batch = "\n\n---\n\n".join(batch_chunks)

            if batch_idx == 0:
                prompt_to_use = reduce_prompt.replace('{combined_map_results}', combined_batch)
            else:
                context_snippet = previous_context[-3000:] if len(previous_context) > 3000 else previous_context
                prompt_to_use = reduce_with_context_prompt.replace('{previous_context}', context_snippet)
                prompt_to_use = prompt_to_use.replace('{combined_map_results}', combined_batch)

            reduce_input = f"""
                    System:
                    {yt_transcript_shortener_system_message}
                    Input:
                    {prompt_to_use}
                """

            print(f"[DEBUG] Running REDUCE batch {batch_idx + 1}...")
            try:
                response = current_model.invoke(reduce_input)
                batch_response_text = response.content
            except Exception as e:
                print(f"[ERROR] Model crashed during REDUCE batch {batch_idx + 1}/{num_batches}: {e}")
                if _has_checkpoint:
                    checkpoint["reduce_retry_counts"][str_batch] = (
                        checkpoint["reduce_retry_counts"].get(str_batch, 0) + 1
                    )
                    _save()
                raise ValueError(f"Model crashed during REDUCE batch {batch_idx + 1}/{num_batches}: {e}")

            print(f"[DEBUG] REDUCE batch {batch_idx + 1} complete: {len(batch_response_text)} chars")
            cleaned_batch, success = remove_thinking_tokens(batch_response_text)
            if not success:
                error_msg = f"Failed to remove thinking tokens from REDUCE batch {batch_idx + 1}/{num_batches}"
                print(f"[ERROR] {error_msg}")
                if _has_checkpoint:
                    checkpoint["reduce_retry_counts"][str_batch] = (
                        checkpoint["reduce_retry_counts"].get(str_batch, 0) + 1
                    )
                    _save()
                raise ValueError(error_msg)

            if _has_checkpoint:
                checkpoint["reduce_results"][str_batch] = cleaned_batch
                _save()

            batch_results.append(cleaned_batch)
            previous_context = cleaned_batch
            print(f"[SUCCESS] REDUCE batch {batch_idx + 1} complete: {len(cleaned_batch)} chars")

        final_output = "\n\n".join(batch_results)
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] All REDUCE batches combined: {len(final_output)} chars")

        # Stage 4 — optional final consolidation
        if len(final_output) > FINAL_CONSOLIDATION_THRESHOLD:
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"Final consolidation needed ({len(final_output)} > {FINAL_CONSOLIDATION_THRESHOLD} chars)"
            )

            # Resume: reuse cached consolidation result
            if _has_checkpoint and checkpoint.get("consolidation_result"):
                final_output = checkpoint["consolidation_result"]
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: final consolidation already complete, skipping")
            else:
                if _has_checkpoint:
                    consol_retries = checkpoint.get("consolidation_retries", 0)
                    if consol_retries >= MAX_RETRIES_PER_STEP:
                        raise ValueError(
                            f"Final consolidation exceeded max retries ({MAX_RETRIES_PER_STEP})."
                        )

                consolidation_input = f"""
                    System:
                    {yt_transcript_shortener_system_message}
                    Input:
                    {reduce_prompt.replace('{combined_map_results}', final_output)}
                """

                print(f"[DEBUG] Running final consolidation...")
                try:
                    response = current_model.invoke(consolidation_input)
                    consolidation_text = response.content
                except Exception as e:
                    print(f"[ERROR] Model crashed during final consolidation: {e}")
                    if _has_checkpoint:
                        checkpoint["consolidation_retries"] = (
                            checkpoint.get("consolidation_retries", 0) + 1
                        )
                        _save()
                    raise ValueError(f"Model crashed during final consolidation: {e}")

                consolidated, success = remove_thinking_tokens(consolidation_text)
                if not success:
                    error_msg = "Failed to remove thinking tokens from final consolidation"
                    print(f"[ERROR] {error_msg}")
                    if _has_checkpoint:
                        checkpoint["consolidation_retries"] = (
                            checkpoint.get("consolidation_retries", 0) + 1
                        )
                        _save()
                    raise ValueError(error_msg)

                if _has_checkpoint:
                    checkpoint["consolidation_result"] = consolidated
                    _save()

                final_output = consolidated
                print(f"[SUCCESS] Final consolidation complete: {len(final_output)} chars")
        else:
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"No final consolidation needed ({len(final_output)} chars < {FINAL_CONSOLIDATION_THRESHOLD})"
            )

    # ------------------------------------------------------------------
    # Stage 5 — persist final output
    # ------------------------------------------------------------------
    if _has_checkpoint:
        checkpoint["final_output"] = final_output
        _save()

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] REDUCE phase complete: {len(final_output)} chars")
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