import time
from openai import OpenAI


THINKING_MODE_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()
        self.max_retries = 5
        self.base_wait_time = 30  # seconds

    def get_text(
        self,
        text,
        model,
        max_tokens=500,
        temperature=0.0,
        top_p=1.00,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        thinking=False,
        extract_answer_only=False,
    ):
        import re

        def extract_answer_letter(content):
            content = str(content).strip()
            if re.fullmatch(r"[A-Ea-e]", content):
                return content.upper()

            final_answer_match = re.search(
                r"(?is)final\s*answer\s*[:\-]?\s*([A-Ea-e])\b",
                content,
            )
            if final_answer_match:
                return final_answer_match.group(1).upper()

            all_letters = re.findall(r"\b([A-Ea-e])\b", content)
            return all_letters[-1].upper() if all_letters else "E"

        messages = [{"role": "user", "content": text}]
        if thinking:
            max_tokens = 12000
            temperature = THINKING_MODE_SAMPLING_PARAMS["temperature"]
            top_p = THINKING_MODE_SAMPLING_PARAMS["top_p"]
            presence_penalty = THINKING_MODE_SAMPLING_PARAMS["presence_penalty"]
            extra = {
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": THINKING_MODE_SAMPLING_PARAMS["top_k"],
                "min_p": THINKING_MODE_SAMPLING_PARAMS["min_p"],
                "repetition_penalty": THINKING_MODE_SAMPLING_PARAMS["repetition_penalty"],
            }
        else:
            # Keep non-thinking fully compatible with endpoints that reject
            # chat_template_kwargs.
            extra = None

        supports_chat_template_kwargs = True
        supports_custom_temperature = True

        for attempt in range(self.max_retries):
            try:
                # Newer models (gpt-5+, o1, o3, etc.) use max_completion_tokens
                # These reasoning models need more tokens (they use tokens for internal reasoning)
                is_reasoning_model = any(prefix in model.lower() for prefix in ['gpt-5', 'o1', 'o3'])

                if is_reasoning_model:
                    # Reasoning models need significantly more tokens
                    # Original max_tokens is multiplied by 10 to account for reasoning overhead
                    adjusted_tokens = max(max_tokens * 10, 100)
                    adjusted_tokens = min(adjusted_tokens, 128000)
                    request_kwargs = dict(
                        model=model,
                        max_completion_tokens=adjusted_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        messages=messages,
                    )
                    if supports_custom_temperature:
                        request_kwargs["temperature"] = temperature
                    if supports_chat_template_kwargs and extra is not None:
                        request_kwargs["extra_body"] = extra
                    response = self.client.chat.completions.create(**request_kwargs)
                else:
                    request_kwargs = dict(
                        model=model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        messages=messages,
                    )
                    if supports_custom_temperature:
                        request_kwargs["temperature"] = temperature
                    if supports_chat_template_kwargs and extra is not None:
                        request_kwargs["extra_body"] = extra
                    response = self.client.chat.completions.create(**request_kwargs)

                # Check if the response has valid data
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices[0]

                    # Handle both message.content and potential refusal
                    if hasattr(first_choice, 'message'):
                        message = first_choice.message
                        
                        # Check for refusal first (newer models)
                        if hasattr(message, 'refusal') and message.refusal:
                            raise Exception(f"Model refused to respond: {message.refusal}")
                        
                        # Get content
                        if hasattr(message, 'content') and message.content:
                            content = str(message.content).strip()

                            if thinking:
                                # Strip everything up to and including </think>
                                content = re.sub(r'^.*?</think>\s*', '', content, flags=re.DOTALL).strip()

                            if extract_answer_only:
                                return extract_answer_letter(content)

                            return content
                        
                        # Handle empty content (usually means hit token limit)
                        if first_choice.finish_reason == 'length':
                            raise Exception(
                                f"Response truncated due to token limit. Increase max_tokens (currently {max_tokens})."
                            )
                        
                        raise Exception(
                            f"Response message exists but no content. Finish reason: {first_choice.finish_reason}"
                        )
                    else:
                        raise Exception(
                            f"Response from OpenAI API does not contain 'message'. Choice: {first_choice}"
                        )
                else:
                    raise Exception(
                        "Response from OpenAI API does not contain "
                        "'choices' or choices list is empty."
                    )

            except Exception as e:
                error_str = str(e)
                if supports_chat_template_kwargs and (
                    "chat_template_kwargs" in error_str
                    or "unknown_parameter" in error_str.lower()
                ):
                    supports_chat_template_kwargs = False
                    print("⚠️ Endpoint does not support chat_template_kwargs. Retrying without it.")
                    continue
                if supports_custom_temperature and (
                    "temperature" in error_str.lower()
                    and "default (1) value is supported" in error_str.lower()
                ):
                    supports_custom_temperature = False
                    print("⚠️ Model only supports default temperature. Retrying without explicit temperature.")
                    continue
                is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
                is_last_attempt = attempt == self.max_retries - 1

                if is_rate_limit and not is_last_attempt:
                    wait_time = self.base_wait_time * (2 ** attempt)
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to create completion with OpenAI API: {str(e)}")