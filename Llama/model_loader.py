import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ModelWrapper:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.hook_handle = None

    def load(self):
        print(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
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

        # 3. Attach to the embedding layer
        # For Phi-3.5 and Llama models, this is usually model.model.embed_tokens
        # Check your specific model architecture if this fails.
        self.hook_handle = self.model.model.embed_tokens.register_forward_hook(noise_hook)
        print(f"Registered Gaussian Noise Hook (alpha={noise_alpha})")
