import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.hook_handle = None

        if model_name == "phi":
            self.model_id = "microsoft/Phi-3.5-mini-instruct"
        elif model_name == "mistral":
            self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        elif model_name == "llama":
            self.model_id = "meta-llama/Llama-2-7b-instruct-v0.1"
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Use 'phi', 'mistral', or 'llama'.")

    def load(self):
        print(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if self.model_name == "phi":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map={"": 0},
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=dtype,
                attn_implementation="sdpa"
            )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.pipe, self.tokenizer

    def register_embedding_noise(self, noise_alpha=0.0):
        """
        Registers a hook to inject Gaussian noise into embeddings.
        noise_alpha: Standard deviation of the noise (e.g., 0.05).
        """
        # 1. Clear existing hook if any
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

        if noise_alpha <= 0:
            return

        # 2. Define the hook function
        def noise_hook(module, args, output):
            # output is the tensor of embeddings: [Batch, Seq, Dim]
            # Create noise tensor on the same device/dtype
            noise = torch.randn_like(output) * noise_alpha
            return output + noise

        # 3. Attach to the embedding layer (works across architectures).
        embed_layer = getattr(getattr(self.model, "model", None), "embed_tokens", None)
        if embed_layer is None:
            embed_layer = self.model.get_input_embeddings()
        self.hook_handle = embed_layer.register_forward_hook(noise_hook)
        print(f"Registered Gaussian Noise Hook (alpha={noise_alpha})")
