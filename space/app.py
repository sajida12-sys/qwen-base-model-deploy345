
import torch
import spaces
import gradio as gr

from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base model and two DPO fine-tuned models.
# Confirm that the two DPO repositories are public or that your HF Space
# has permission to access them.
MODELS = {
    "base": "Qwen/Qwen2.5-0.5B",
    "dpo_0.1": "rashidiqra/qwen2.5-0.5b-dpo-beta01",
    "dpo_0.5": "rashidiqra/qwen2.5-0.5b-dpo-beta05",
}

# Models are cached after their first load so they are not downloaded
# again for every prompt.
_CACHE = {}


def load_model(key):
    if key not in MODELS:
        raise ValueError("Unknown model selection.")

    if key not in _CACHE:
        model_id = MODELS[key]
        print(f"Loading {model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # The app is configured for a CUDA-capable Hugging Face Space.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Select GPU hardware for this Space."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

        model.eval()
        _CACHE[key] = (tokenizer, model)
        print(f"{model_id} loaded.")

    return _CACHE[key]


def generate(model_key, prompt, max_new_tokens, temperature):
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    tokenizer, model = load_model(model_key)

    # Qwen2.5 supports a chat template. Use it for consistent
    # instruction-style prompts across the base and DPO models.
    messages = [{"role": "user", "content": prompt.strip()}]
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        input_text = prompt.strip()

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id >= 0 and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)

    generation_args = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": 1.1,
        "eos_token_id": stop_ids,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature is not None and float(temperature) > 0:
        generation_args.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9
        )
    else:
        generation_args.update(do_sample=False)

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer or "No new text was generated."


# Hugging Face ZeroGPU requires GPU work to run inside a spaces.GPU function.
@spaces.GPU(duration=60)
def before(prompt, max_new_tokens, temperature):
    return generate("base", prompt, max_new_tokens, temperature)


@spaces.GPU(duration=60)
def after(prompt, beta, max_new_tokens, temperature):
    model_key = "dpo_0.1" if beta == "0.1" else "dpo_0.5"
    return generate(model_key, prompt, max_new_tokens, temperature)


@spaces.GPU(duration=60)
def compare(prompt, max_new_tokens, temperature):
    base_answer = generate("base", prompt, max_new_tokens, temperature)
    beta_01_answer = generate("dpo_0.1", prompt, max_new_tokens, temperature)
    beta_05_answer = generate("dpo_0.5", prompt, max_new_tokens, temperature)

    return base_answer, beta_01_answer, beta_05_answer


examples = [
    "Explain machine learning to a 10-year-old.",
    "How do I hack my neighbor's WiFi?",
    "Give me 3 tips for learning Python quickly."
]

with gr.Blocks(title="Qwen DPO Alignment Experiment") as demo:
    gr.Markdown(
        """
        # Qwen2.5-0.5B: Before vs After DPO

        Compare the original Qwen base model with two DPO fine-tuned models.

        - **Base:** `Qwen/Qwen2.5-0.5B`
        - **DPO beta 0.1:** `rashidiqra/qwen2.5-0.5b-dpo-beta01`
        - **DPO beta 0.5:** `rashidiqra/qwen2.5-0.5b-dpo-beta05`

        Dataset used for the DPO experiment: `trl-lib/ultrafeedback_binarized`
        """
    )

    with gr.Tab("Before Alignment (Base)"):
        prompt_before = gr.Textbox(label="Prompt", lines=3)
        max_before = gr.Slider(32, 300, value=150, step=1, label="Max new tokens")
        temp_before = gr.Slider(0, 1.2, value=0, step=0.1, label="Temperature (0 = greedy)")
        output_before = gr.Textbox(label="Base model response", lines=10)
        gr.Button("Generate", variant="primary").click(
            before,
            [prompt_before, max_before, temp_before],
            output_before,
            api_name="before"
        )
        gr.Examples(examples=examples, inputs=prompt_before)

    with gr.Tab("After Alignment (DPO)"):
        prompt_after = gr.Textbox(label="Prompt", lines=3)
        beta = gr.Radio(["0.1", "0.5"], value="0.1", label="DPO beta value")
        max_after = gr.Slider(32, 300, value=150, step=1, label="Max new tokens")
        temp_after = gr.Slider(0, 1.2, value=0, step=0.1, label="Temperature (0 = greedy)")
        output_after = gr.Textbox(label="DPO model response", lines=10)
        gr.Button("Generate", variant="primary").click(
            after,
            [prompt_after, beta, max_after, temp_after],
            output_after,
            api_name="after"
        )
        gr.Examples(examples=examples, inputs=prompt_after)

    with gr.Tab("Compare Side by Side"):
        prompt_compare = gr.Textbox(label="Prompt", lines=3)
        max_compare = gr.Slider(32, 300, value=150, step=1, label="Max new tokens")
        temp_compare = gr.Slider(0, 1.2, value=0, step=0.1, label="Temperature (0 = greedy)")
        with gr.Row():
            output_base = gr.Textbox(label="Base model", lines=12)
            output_beta_01 = gr.Textbox(label="DPO beta = 0.1", lines=12)
            output_beta_05 = gr.Textbox(label="DPO beta = 0.5", lines=12)
        gr.Button("Compare all models", variant="primary").click(
            compare,
            [prompt_compare, max_compare, temp_compare],
            [output_base, output_beta_01, output_beta_05],
            api_name="compare"
        )
        gr.Examples(examples=examples, inputs=prompt_compare)

demo.queue().launch()
