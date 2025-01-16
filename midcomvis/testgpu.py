import torch
import torch
import torchvision
from ultralytics import YOLO

# print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("Is CUDA available?", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device:", torch.cuda.get_device_name(0))

# torch.set_default_device("cuda")

# model = YOLO("yolov8n.pt")
# print("Using CUDA:", model.device)  # Should display 'cuda'


# Test CUDA availability
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# Test tensor operation
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 3, device=device)
print("Tensor on device:", x)


print(f"Available GPU memory: {torch.cuda.memory_allocated()} bytes")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory} bytes")


torch.cuda.empty_cache()
tensor = torch.empty(10000, 10000, device="cuda")
print(tensor)
