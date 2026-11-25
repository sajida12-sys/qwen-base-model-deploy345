
import torch
import spaces
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading Qwen pretrained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# ZeroGPU model placement
model = model.to("cuda")
model.eval()

print("Qwen model loaded successfully!")


@spaces.GPU(duration=60)
def generate_response(prompt, max_tokens):

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    try:
        inputs = tokenizer(
            prompt.strip(),
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
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

        return answer.strip() or "No new text was generated."

    except Exception as e:
        return f"Generation error: {str(e)}"


with gr.Blocks(title="Qwen Base AI") as demo:

    gr.Markdown(
        """
        # Qwen Base AI

        Text generation using the original
        pretrained Qwen/Qwen2.5-0.5B model.

        This application demonstrates text continuation
        using a pretrained base language model.
        """
    )

    prompt = gr.Textbox(
        label="Enter Your Prompt",
        placeholder="Enter a sentence or text to continue...",
        lines=5
    )

    max_tokens = gr.Slider(
        minimum=50,
        maximum=300,
        value=150,
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
