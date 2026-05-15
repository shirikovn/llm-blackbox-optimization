import ollama

from src.prompts.system import SYSTEM_PROMPT
from src.prompts.serializers import build_prompt
from src.utils.parsing import parse_vector


class LLMOptimizer:
    def __init__(
        self,
        logger,
        model,
        temperature=0.0,
        history_size=5,
        use_gradient=True,
    ):

        self.logger = logger

        self.model = model

        self.temperature = temperature

        self.history_size = history_size
        self.use_gradient = use_gradient

    def step(
        self,
        history,
        step_id,
    ):

        prompt = build_prompt(
            history=history,
            use_gradient=self.use_gradient,
            history_size=self.history_size,
        )

        self.logger.log_prompt(
            step_id,
            prompt,
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": self.temperature,
            },
        )

        text = response["message"]["content"]

        self.logger.log_response(
            step_id,
            text,
        )

        return parse_vector(text)
