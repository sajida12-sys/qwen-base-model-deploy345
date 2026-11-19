import spaces
import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading original Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Required for Hugging Face ZeroGPU
model = model.to("cuda")
model.eval()

print("Qwen model loaded successfully!")


@spaces.GPU(duration=60)
def generate_response(prompt, max_tokens):

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0, inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


with gr.Blocks(title="Qwen Base AI") as demo:

    gr.Markdown(
        """
        # Qwen Base AI

        Text generation using the original
        Qwen/Qwen2.5-0.5B pretrained model.
        """
    )

    prompt = gr.Textbox(
        label="Enter Your Prompt",
        placeholder="Ask your question...",
        lines=5
    )

    max_tokens = gr.Slider(
        minimum=50,
        maximum=300,
        value=180,
        step=10,
        label="Maximum New Tokens"
    )

    generate_button = gr.Button(
        "Generate Response",
        variant="primary"
    )

    output = gr.Textbox(
        label="Generated Response",
        lines=10
    )

    generate_button.click(
        fn=generate_response,
        inputs=[prompt, max_tokens],
        outputs=output
    )

demo.queue().launch()
