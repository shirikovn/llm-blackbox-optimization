from src.prompts.serializers import build_prompt
from src.utils.parsing import parse_vector


class InteractiveLLMOptimizer:
    def __init__(
        self,
        logger,
        history_size=5,
        use_gradient=True,
    ):

        self.logger = logger

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
            step=step_id,
            prompt=prompt,
        )

        print("\n" + "=" * 80)
        print(f"STEP {step_id}")
        print("=" * 80)

        print("\nCOPY THIS PROMPT:\n")

        print(prompt)

        print("\n" + "=" * 80)

        print("\nPaste model response below.")

        print("Finish with ENTER twice.\n")

        lines = []

        while True:
            line = input()

            if line == "":
                break

            lines.append(line)

        response = "\n".join(lines)

        self.logger.log_response(
            step=step_id,
            response=response,
        )

        print("\nRAW RESPONSE:\n")
        print(response)

        try:
            x_next = parse_vector(response)

        except Exception as e:
            self.logger.log_error(
                {
                    "step": step_id,
                    "type": "parse_error",
                    "response": response,
                    "error": str(e),
                }
            )

            raise

        print("\nPARSED:\n")
        print(x_next)

        return x_next
