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
    ):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    messages=[{"role": "user", "content": text}],
                )

                # Check if the response has valid data
                if response.choices and len(response.choices) > 0:
                    first_choice = response.choices[0]

                    if first_choice.message and first_choice.message.content:
                        return str(first_choice.message.content)
                    else:
                        raise Exception(
                            "Response from OpenAI API does not "
                            "contain 'message' or 'content'."
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
