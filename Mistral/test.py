from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map={"": 0}, torch_dtype=torch.float16,
    trust_remote_code=True, attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Check what layer 0 forward hook receives
outputs = []
def hook(module, inp, out):
    outputs.append((type(out), len(out) if isinstance(out, tuple) else "not tuple",
                    out[0].shape if isinstance(out, tuple) else out.shape))
    return out

h = model.model.layers[0].register_forward_hook(hook)
inp = tokenizer("Hello world", return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**inp, output_hidden_states=False)
h.remove()
print("Layer 0 output:", outputs[0])

