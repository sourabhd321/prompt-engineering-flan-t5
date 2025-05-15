def generate_prompts(text, few_shot_examples=None):
    def format_example(e):
        return f"Article: {e['text']}\nSummary: {e['summary']}"

    few_shot_block = ""
    if few_shot_examples:
        few_shot_block = "\n\n".join([format_example(e) for e in few_shot_examples]) + "\n\n"

    return {
        "zero_shot": text,
        "one_shot": (
            format_example(few_shot_examples[0]) + f"\n\nNow, summarize: {text}"
        ) if few_shot_examples else text,
        "few_shot": (
            few_shot_block + f"Now, summarize: {text}"
        ),
        "chain_of_thought": (
            "Let's think step by step. What is the key topic and outcome?\n" + text
        ),
        "role_based": (
            "You are a professional news editor. " + text
        ),
        "instruction_tuned": (
            "Summarize the following news article in one line: " + text
        ),
        "generative_config": (
            "Summarize this news in a persuasive and professional tone, using no more than 20 words. "
            "Avoid jargon. Keep it impactful:\n" + text
        )
    }
