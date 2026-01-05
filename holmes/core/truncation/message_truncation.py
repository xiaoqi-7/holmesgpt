import logging

from holmes.common.env_vars import MAX_OUTPUT_TOKEN_RESERVATION
from holmes.core.models import TruncationMetadata, TruncationResult
from holmes.utils import sentry_helper


TRUNCATION_NOTICE = "\n\n[TRUNCATED]"


def _truncate_tool_message(
    msg: dict, allocated_space: int, needed_space: int
) -> TruncationMetadata:
    msg_content = msg["content"]
    tool_call_id = msg.get("tool_call_id")
    tool_name = msg.get("name")

    # Ensure the indicator fits in the allocated space
    if allocated_space > len(TRUNCATION_NOTICE):
        original = msg_content if isinstance(msg_content, str) else str(msg_content)
        msg["content"] = (
            original[: allocated_space - len(TRUNCATION_NOTICE)] + TRUNCATION_NOTICE
        )
        end_index = allocated_space - len(TRUNCATION_NOTICE)
    else:
        msg["content"] = TRUNCATION_NOTICE[:allocated_space]
        end_index = allocated_space

    msg.pop("token_count", None)  # Remove token_count if present
    logging.info(
        f"Truncating tool message '{tool_name}' from {needed_space} to {allocated_space} tokens"
    )
    truncation_metadata = TruncationMetadata(
        tool_call_id=tool_call_id,
        start_index=0,
        end_index=end_index,
        tool_name=tool_name,
        original_token_count=needed_space,
    )
    return truncation_metadata


# TODO: I think there's a bug here because we don't account for the 'role' or json structure like '{...}' when counting tokens
# However, in practice it works because we reserve enough space for the output tokens that the minor inconsistency does not matter
# We should fix this in the future
# TODO: we truncate using character counts not token counts - this means we're overly agressive with truncation - improve it by considering
# token truncation and not character truncation
def truncate_messages_to_fit_context(
    messages: list, max_context_size: int, maximum_output_token: int, count_tokens_fn
) -> TruncationResult:
    """Truncate tool messages to fit within context limits."""

    messages_except_tools = [
        message for message in messages if message.get("role") != "tool"
    ]
    tokens = count_tokens_fn(messages_except_tools)
    message_size_without_tools = tokens.total_tokens

    tool_call_messages = [
        message for message in messages if message.get("role") == "tool"
    ]

    reserved_for_output_tokens = min(maximum_output_token, MAX_OUTPUT_TOKEN_RESERVATION)
    if message_size_without_tools >= (max_context_size - reserved_for_output_tokens):
        logging.error(
            f"The combined size of system_prompt and user_prompt ({message_size_without_tools} tokens) exceeds the model's context window for input."
        )
        raise Exception(
            f"The combined size of system_prompt and user_prompt ({message_size_without_tools} tokens) exceeds the maximum context size of {max_context_size - reserved_for_output_tokens} tokens available for input."
        )

    if len(tool_call_messages) == 0:
        return TruncationResult(truncated_messages=messages, truncations=[])

    available_space = (
        max_context_size - message_size_without_tools - reserved_for_output_tokens
    )
    remaining_space = available_space
    tool_call_messages.sort(
        key=lambda x: count_tokens_fn(
            [{"role": "tool", "content": x.get("content")}]
        ).total_tokens
    )

    truncations = []

    for i, msg in enumerate(tool_call_messages):
        remaining_tools = len(tool_call_messages) - i
        max_allocation = remaining_space // remaining_tools
        needed_space = count_tokens_fn(
            [{"role": "tool", "content": msg.get("content")}]
        ).total_tokens
        allocated_space = min(needed_space, max_allocation)

        if needed_space > allocated_space:
            truncation_metadata = _truncate_tool_message(msg, allocated_space, needed_space)
            truncations.append(truncation_metadata)

        remaining_space -= allocated_space

    if truncations:
        sentry_helper.capture_tool_truncations(truncations)

    return TruncationResult(truncated_messages=messages, truncations=truncations)
