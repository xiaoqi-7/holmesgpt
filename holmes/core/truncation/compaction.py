import json
import logging
from typing import Any, Optional

import litellm
from litellm.types.utils import ModelResponse

from holmes.core.llm import LLM
from holmes.core.truncation.message_truncation import truncate_messages_to_fit_context
from holmes.plugins.prompts import load_and_render_prompt

TRUNCATION_NOTICE_FOR_COMPACTION = "\n\n[TRUNCATED_FOR_COMPACTION]"

def _is_context_window_exceeded_error(err: Exception) -> bool:
    # LiteLLM/vLLM providers wrap this in multiple exception types.
    if err.__class__.__name__ == "ContextWindowExceededError":
        return True
    return "ContextWindowExceeded" in str(err)


def _reserved_output_tokens_for_compaction(llm: LLM) -> int:
    # Compaction output should be relatively small.
    try:
        return min(2048, llm.get_maximum_output_token())
    except Exception:
        return 2048


def _safe_stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _prepare_messages_for_compaction(messages: list[dict]) -> list[dict]:
    """Prepare messages for compaction calls.

    Some OpenAI-compatible servers (notably certain vLLM deployments) validate and JSON-parse
    `tool_calls[].function.arguments` in *input* messages. If those strings contain sequences like
    `\|` (common in shell regex), they are invalid JSON escapes and can trigger:
    `Invalid \escape`.

    To make compaction resilient across providers, we strip tool/function-call structures and
    pass only plain role+content messages.
    """

    prepared: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = _safe_stringify_content(msg.get("content"))

        # If the assistant message has tool_calls/function_call but no textual content, preserve a textual hint.
        if (role == "assistant") and (not content.strip()):
            if msg.get("tool_calls") is not None:
                content = "Tool calls were made."
            elif msg.get("function_call") is not None:
                content = "A function call was made."

        if role == "tool":
            tool_name = msg.get("name") or msg.get("tool_name")
            tool_call_id = msg.get("tool_call_id")
            header = "[TOOL]"
            if tool_name:
                header = f"[TOOL {tool_name}]"
            if tool_call_id:
                header = f"{header} (tool_call_id={tool_call_id})"
            prepared.append(
                {
                    "role": "assistant",
                    "content": f"{header}\n{content}",
                }
            )
            continue

        prepared.append(
            {
                "role": role or "user",
                "content": content,
            }
        )

    return prepared


def _shrink_messages_to_fit_compaction_window(
    llm: LLM,
    messages: list[dict],
    max_context_size: int,
    reserved_output_tokens: int,
) -> list[dict]:
    def fits(msgs: list[dict]) -> bool:
        return (
            llm.count_tokens(messages=msgs).total_tokens + reserved_output_tokens
        ) <= max_context_size

    if fits(messages):
        return messages

    # Prefer truncating tool messages fairly before dropping conversational context.
    try:
        messages = truncate_messages_to_fit_context(
            messages=messages,
            max_context_size=max_context_size,
            maximum_output_token=reserved_output_tokens,
            count_tokens_fn=llm.count_tokens,
        ).truncated_messages
        if fits(messages):
            return messages
    except Exception:
        pass

    # Drop oldest messages until it fits (keep most recent context).
    while len(messages) > 2 and not fits(messages):
        messages.pop(0)

    if fits(messages):
        return messages

    # Hard-truncate the largest string message as a last resort.
    for _ in range(12):
        if fits(messages):
            return messages

        candidate_index = None
        candidate_len = -1
        for idx, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if len(content) > candidate_len:
                candidate_len = len(content)
                candidate_index = idx

        if candidate_index is None:
            return messages

        msg = messages[candidate_index]
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < 50:
            return messages

        current_tokens = llm.count_tokens(messages=messages).total_tokens
        target_input_budget = max(1, max_context_size - reserved_output_tokens)
        shrink_ratio = min(0.9, target_input_budget / max(1, current_tokens))
        new_len = max(200, int(len(content) * shrink_ratio))
        msg["content"] = (
            content[: max(0, new_len - len(TRUNCATION_NOTICE_FOR_COMPACTION))]
            + TRUNCATION_NOTICE_FOR_COMPACTION
        )

    return messages


def strip_system_prompt(
    conversation_history: list[dict],
) -> tuple[list[dict], Optional[dict]]:
    if not conversation_history:
        return conversation_history, None
    first_message = conversation_history[0]
    if first_message and first_message.get("role") == "system":
        return conversation_history[1:], first_message
    return conversation_history[:], None


def find_last_user_prompt(conversation_history: list[dict]) -> Optional[dict]:
    if not conversation_history:
        return None
    last_user_prompt: Optional[dict] = None
    for message in conversation_history:
        if message.get("role") == "user":
            last_user_prompt = message
    return last_user_prompt


def compact_conversation_history(
    original_conversation_history: list[dict], llm: LLM
) -> list[dict]:
    """
    The compacted conversation history contains:
      1. Original system prompt, uncompacted (if present)
      2. Last user prompt, uncompacted (if present)
      3. Compacted conversation history (role=assistant)
      4. Compaction message (role=system)
    """
    conversation_history, system_prompt_message = strip_system_prompt(
        original_conversation_history
    )
    compaction_instructions = load_and_render_prompt(
        prompt="builtin://conversation_history_compaction.jinja2", context={}
    )
    conversation_history.append({"role": "user", "content": compaction_instructions})

    # Ensure the compaction request fits the model context window.
    reserved_output_tokens = _reserved_output_tokens_for_compaction(llm)
    max_context_size = llm.get_context_window_size()
    try:
        original_tokens = llm.count_tokens(messages=conversation_history).total_tokens
        conversation_history = _shrink_messages_to_fit_compaction_window(
            llm=llm,
            messages=conversation_history,
            max_context_size=max_context_size,
            reserved_output_tokens=reserved_output_tokens,
        )
        shrunk_tokens = llm.count_tokens(messages=conversation_history).total_tokens
        if shrunk_tokens < original_tokens:
            logging.info(
                "Shrunk compaction input from %s to %s tokens (reserve=%s, window=%s)",
                original_tokens,
                shrunk_tokens,
                reserved_output_tokens,
                max_context_size,
            )
    except Exception as e:
        logging.warning("Failed to pre-shrink compaction input: %s", e)

    # Strip tool_calls/function_call and normalize content to avoid provider-side JSON parsing errors.
    compaction_messages = _prepare_messages_for_compaction(conversation_history)

    # Set modify_params to handle providers like Anthropic that require tools
    # when conversation history contains tool calls
    original_modify_params = litellm.modify_params
    try:
        litellm.modify_params = True  # necessary when using anthropic
        try:
            response: ModelResponse = llm.completion(
                messages=compaction_messages, drop_params=True
            )  # type: ignore
        except Exception as e:
            # Retry once on context-window issues with more aggressive shrinking.
            if _is_context_window_exceeded_error(e):
                logging.warning(
                    "Context window exceeded during compaction; retrying with aggressive shrinking"
                )
                reserved_retry = min(512, reserved_output_tokens)
                conversation_history = _shrink_messages_to_fit_compaction_window(
                    llm=llm,
                    messages=conversation_history,
                    max_context_size=max_context_size,
                    reserved_output_tokens=reserved_retry,
                )
                compaction_messages = _prepare_messages_for_compaction(
                    conversation_history
                )
                response = llm.completion(
                    messages=compaction_messages, drop_params=True
                )  # type: ignore
            else:
                raise
    finally:
        litellm.modify_params = original_modify_params
    response_message = None
    if (
        response
        and response.choices
        and response.choices[0]
        and response.choices[0].message  # type:ignore
    ):
        response_message = response.choices[0].message  # type:ignore
    else:
        logging.error(
            "Failed to compact conversation history. Unexpected LLM's response for compaction"
        )
        return original_conversation_history

    compacted_conversation_history: list[dict] = []
    if system_prompt_message:
        compacted_conversation_history.append(system_prompt_message)

    last_user_prompt = find_last_user_prompt(original_conversation_history)
    if last_user_prompt:
        compacted_conversation_history.append(last_user_prompt)

    compacted_conversation_history.append(
        response_message.model_dump(
            exclude_defaults=True, exclude_unset=True, exclude_none=True
        )
    )

    compacted_conversation_history.append(
        {
            "role": "system",
            "content": "The conversation history has been compacted to preserve available space in the context window. Continue.",
        }
    )
    return compacted_conversation_history
