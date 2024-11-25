import torch
import torchaudio
import soundfile as sf

print("PyTorch version:", torch.__version__)
print("TorchAudio version:", torchaudio.__version__)
print("MPS (Metal) available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

x = torch.randn(2, 3)
print("\nTest tensor:")
print(x)

if torch.backends.mps.is_available():
    x_mps = x.to('mps')
    print("\nMPS tensor:")
    print(x_mps)
