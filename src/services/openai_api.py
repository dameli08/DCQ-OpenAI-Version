import time
from openai import OpenAI


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
    ):
        import re
        messages = [{"role": "user", "content": text}]
        if thinking:
            # thinking_budget is a Qwen3 hard cap: once that many reasoning tokens are used,
            # the model is forced to emit </think> and proceed to the answer.
            # max_tokens must be > thinking_budget so there is always room for the answer letter.
            extra = {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 1024}}
            max_tokens = 1200  # 1024 thinking budget + 176 buffer for the answer letter
        else:
            extra = {}

        for attempt in range(self.max_retries):
            try:
                # Newer models (gpt-5+, o1, o3, etc.) use max_completion_tokens
                # These reasoning models need more tokens (they use tokens for internal reasoning)
                is_reasoning_model = any(prefix in model.lower() for prefix in ['gpt-5', 'o1', 'o3'])

                if is_reasoning_model:
                    # Reasoning models need significantly more tokens
                    # Original max_tokens is multiplied by 10 to account for reasoning overhead
                    adjusted_tokens = max(max_tokens * 10, 100)
                    response = self.client.chat.completions.create(
                        model=model,
                        max_completion_tokens=adjusted_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        messages=messages,
                        extra_body=extra,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        messages=messages,
                        extra_body=extra,
                    )

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
                            content = str(message.content)
                            # Strip thinking block (model may omit opening <think> tag,
                            # so strip everything up to and including </think>)
                            content = re.sub(r'^.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                            # Fallback: </think> should always be present when thinking_budget
                            # < max_tokens, but if it's somehow missing, retry once with
                            # thinking still enabled so the model always reasons properly.
                            if thinking and not re.match(r'^[A-E]$', content):
                                print(f"⚠️ </think> missing in thinking response — retrying with thinking enabled.")
                                return self.get_text(
                                    text=text, model=model, max_tokens=max_tokens,
                                    temperature=temperature, top_p=top_p,
                                    frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty, thinking=True,
                                )
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
                is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
                is_last_attempt = attempt == self.max_retries - 1

                if is_rate_limit and not is_last_attempt:
                    wait_time = self.base_wait_time * (2 ** attempt)
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to create completion with OpenAI API: {str(e)}")
