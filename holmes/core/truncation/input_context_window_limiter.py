import logging
from typing import Any, Optional
from pydantic import BaseModel
import sentry_sdk
from holmes.common.env_vars import (
    ENABLE_CONVERSATION_HISTORY_COMPACTION,
)
from holmes.core.llm import (
    LLM,
    TokenCountMetadata,
    get_context_window_compaction_threshold_pct,
)
from holmes.core.models import TruncationMetadata, TruncationResult
from holmes.core.truncation.compaction import compact_conversation_history
from holmes.core.truncation.message_truncation import truncate_messages_to_fit_context
from holmes.utils import sentry_helper
from holmes.utils.stream import StreamEvents, StreamMessage


class ContextWindowLimiterOutput(BaseModel):
    metadata: dict
    messages: list[dict]
    events: list[StreamMessage]
    max_context_size: int
    maximum_output_token: int
    tokens: TokenCountMetadata
    conversation_history_compacted: bool


@sentry_sdk.trace
def limit_input_context_window(
    llm: LLM, messages: list[dict], tools: Optional[list[dict[str, Any]]]
) -> ContextWindowLimiterOutput:
    events = []
    metadata = {}
    initial_tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    max_context_size = llm.get_context_window_size()
    maximum_output_token = llm.get_maximum_output_token()
    conversation_history_compacted = False
    if ENABLE_CONVERSATION_HISTORY_COMPACTION and (
        initial_tokens.total_tokens + maximum_output_token
    ) > (max_context_size * get_context_window_compaction_threshold_pct() / 100):
        compacted_messages = compact_conversation_history(
            original_conversation_history=messages, llm=llm
        )
        compacted_tokens = llm.count_tokens(compacted_messages, tools=tools)
        compacted_total_tokens = compacted_tokens.total_tokens

        if compacted_total_tokens < initial_tokens.total_tokens:
            messages = compacted_messages
            compaction_message = f"The conversation history has been compacted from {initial_tokens.total_tokens} to {compacted_total_tokens} tokens"
            logging.info(compaction_message)
            conversation_history_compacted = True
            events.append(
                StreamMessage(
                    event=StreamEvents.CONVERSATION_HISTORY_COMPACTED,
                    data={
                        "content": compaction_message,
                        "messages": compacted_messages,
                        "metadata": {
                            "initial_tokens": initial_tokens.total_tokens,
                            "compacted_tokens": compacted_total_tokens,
                        },
                    },
                )
            )
            events.append(
                StreamMessage(
                    event=StreamEvents.AI_MESSAGE,
                    data={"content": compaction_message},
                )
            )
        else:
            logging.debug(
                f"Failed to reduce token count when compacting conversation history. Original tokens:{initial_tokens.total_tokens}. Compacted tokens:{compacted_total_tokens}"
            )

    tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    if (tokens.total_tokens + maximum_output_token) > max_context_size:
        # Compaction was not sufficient. Truncating messages.
        truncated_res = truncate_messages_to_fit_context(
            messages=messages,
            max_context_size=max_context_size,
            maximum_output_token=maximum_output_token,
            count_tokens_fn=llm.count_tokens,
        )
        metadata["truncations"] = [t.model_dump() for t in truncated_res.truncations]
        messages = truncated_res.truncated_messages

        # recount after truncation
        tokens = llm.count_tokens(messages=messages, tools=tools)  # type: ignore
    else:
        metadata["truncations"] = []

    return ContextWindowLimiterOutput(
        events=events,
        messages=messages,
        metadata=metadata,
        max_context_size=max_context_size,
        maximum_output_token=maximum_output_token,
        tokens=tokens,
        conversation_history_compacted=conversation_history_compacted,
    )
