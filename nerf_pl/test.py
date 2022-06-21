#%%
import pickle
import torch
import matplotlib.pyplot as plt
with open('test_rgbs.pkl',"rb") as f:
    rgb = pickle.load(f).cpu()
    rgb = rgb.view(50,50,3)
    plt.imshow(rgb)

#%%
with open('test_results.pkl',"rb") as f:
    rgb = pickle.load(f)
    rgb = rgb['rgb_fine'].detach().cpu()
    rgb = rgb.view(50,50,3)
    plt.imshow(rgb)

#%%

# %%
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
a = Image.open("/root/nerf_pl/results/llff/room/000.png")
image = preprocess(Image.open("/root/nerf_pl/results/llff/room/000.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a room", "a tv", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# %%
import torchvision
import torchvision.transforms as transforms
transforms.InterpolationMode.nearest
# %%
