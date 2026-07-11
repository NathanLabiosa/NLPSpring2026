# compat.py — Compatibility shims for transformers API changes.
# Import this early (it is imported by train_lrd_stabilizer.py and eval scripts).

# transformers >= 4.46 renamed DynamicCache.get_max_length → get_seq_length.
# Phi-3.5's downloaded modeling_phi3.py still calls get_max_length, so
# patch it back in when missing.
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = DynamicCache.get_seq_length
except Exception:
    pass
