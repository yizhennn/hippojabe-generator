from diffusers import DiffusionPipeline
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
pipeline = DiffusionPipeline.from_pretrained("C:\hippojabe-generator\tunedmodel", torch_dtype=torch.float16, use_safetensors=True).to(device)
image = pipeline("show a cute dog in the style of hippojabe", num_inference_steps=100, guidance_scale=9).images[0]
image.save("./outputimages/img0.png")