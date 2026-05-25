import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from stabilizer import DepthWiseStabilizer


class ModelWrapper:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.hook_handle = None
        self.stabilizer_hook_handle = None
        self.stabilizer = None

    def load(self):
        print(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"
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
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

        if noise_alpha <= 0:
            return

        def noise_hook(module, args, output):
            noise = torch.randn_like(output) * noise_alpha
            return output + noise

        self.hook_handle = self.model.model.embed_tokens.register_forward_hook(noise_hook)
        print(f"Registered Gaussian Noise Hook (alpha={noise_alpha})")

    def load_stabilizer(self, checkpoint_path: str, inject_layer: int = 16):
        """
        Load a trained DepthWiseStabilizer from a checkpoint and inject it
        as a forward hook at the specified transformer layer.

        The stabilizer runs on the OUTPUT of the target layer, correcting
        the hidden states before they flow into the next layer's attention.

        Args:
            checkpoint_path (str): Path to stabilizer_final.pt (or any checkpoint).
            inject_layer (int):    Transformer layer index to inject at.
                                   Should match the layer used during training.
        """
        # Clear any existing stabilizer hook
        if self.stabilizer_hook_handle:
            self.stabilizer_hook_handle.remove()
            self.stabilizer_hook_handle = None

        # Load checkpoint
        device = next(self.model.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize stabilizer with correct hidden dim
        hidden_dim = self.model.config.hidden_size
        self.stabilizer = DepthWiseStabilizer(hidden_dim=hidden_dim)
        self.stabilizer.load_state_dict(checkpoint["stabilizer_state_dict"])
        self.stabilizer.eval()
        # Run in float32 even if backbone is float16 (avoids precision issues in the small MLP)
        self.stabilizer = self.stabilizer.to(device).to(torch.float32)

        print(f"Loaded stabilizer from: {checkpoint_path}")
        print(f"  Inject layer: {inject_layer}  |  Gate α: {self.stabilizer.alpha.item():.4f}")

        # Register the hook on the target transformer layer
        target_layer = self.model.model.layers[inject_layer]

        def stabilizer_hook(module, args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output

            original_dtype = hidden_states.dtype
            with torch.no_grad():
                corrected, _ = self.stabilizer(hidden_states.to(torch.float32))  # unpack (h, alpha)
            corrected = corrected.to(original_dtype)

            if isinstance(output, tuple):
                return (corrected,) + output[1:]
            return corrected

        self.stabilizer_hook_handle = target_layer.register_forward_hook(stabilizer_hook)
        print(f"Stabilizer hook registered on layer {inject_layer}.")

    def remove_stabilizer(self):
        """Detach the stabilizer hook (useful for ablation experiments)."""
        if self.stabilizer_hook_handle:
            self.stabilizer_hook_handle.remove()
            self.stabilizer_hook_handle = None
            print("Stabilizer hook removed.")
