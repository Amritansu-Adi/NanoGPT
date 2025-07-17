# Add this at the start of your script
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
