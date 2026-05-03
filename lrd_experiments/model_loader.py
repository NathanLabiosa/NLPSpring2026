import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class ModelWrapper:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.hook_handle = None  # for embedding noise experiments

    def load(self):
        print(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # print("set pad token to eos")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa"  # tried "flash_attention_2" but got OOM
        )
        # print("model loaded to gpu")

        # create pipeline for easier generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # print("pipeline created")
        return self.pipe, self.tokenizer

    def register_embedding_noise(self, noise_alpha=0.0):
        """
        Registers a hook to inject Gaussian noise into embeddings.
        noise_alpha: Standard deviation of the noise (e.g., 0.05).
        This was used for some early robustness experiments but we ended up
        not using it for the final paper.
        """
        # clear existing hook if any
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

        if noise_alpha <= 0:
            return  # no noise
        # print(f"registering noise hook alpha={noise_alpha}")

        # define the hook function
        def noise_hook(module, args, output):
            # output is the tensor of embeddings: [Batch, Seq, Dim]
            noise = torch.randn_like(output) * noise_alpha
            # print(f"Injecting noise with std={noise_alpha}")
            return output + noise

        # attach to the embedding layer
        # for Phi-3.5 and Llama models, this is usually model.model.embed_tokens
        # TODO: might need to adjust for other architectures
        self.hook_handle = self.model.model.embed_tokens.register_forward_hook(noise_hook)
        print(f"Registered Gaussian Noise Hook (alpha={noise_alpha})")
