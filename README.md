# hippojabe-generator
A fine-tuned model designed to generate images in the style of 'hippojabe,' based on a DreamBooth fine-tuned version of Stable Diffusion v1.5.

Using python 3.11.9 <br>
Running the following command in Windows PowerShell

# 0. Using Model: Stable Diffusion v1-5
```
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 
```
# 1.Using Library: Diffusers and Method: DreamBooth
#### The diffusers library is a popular open-source Python library developed by Hugging Face. It provides tools for working with diffusion models, which are a type of generative model used in machine learning, especially for creating images, audio, and other types of data from noise. DreamBooth is a technique developed by researchers at Google Research and Boston University to fine-tune text-to-image diffusion models like Stable Diffusion. The main goal of DreamBooth is to personalize these models so they can generate images that are not only faithful to specific subjects but also retain the ability to generate creative and diverse images in various contexts or styles.
```
git clone https://github.com/huggingface/diffusers

cd diffusers
pip install .

cd examples/dreambooth
pip install -r requirements.txt
```
# 2. Set up a default configuration for Accelerate
#### Accelerate is a library developed by Hugging Face designed to simplify and optimize the training and inference of machine learning models, particularly in distributed and multi-device environments. It provides tools to make it easier to run training and inference tasks across different hardware setups, such as CPUs, GPUs, and TPUs.

```
accelerate config default
```
# 3. Adding images in ./inputdata 

# 4. Launch the DreamBooth training process using the Accelerate library
```
accelerate launch ./diffusers/examples/dreambooth/train_dreambooth.py  \
--pretrained_model_name_or_path="C:\hippojabe-generator\stable-diffusion-v1-5"  \
--instance_data_dir="C:\hippojabe-generator\InputImages" \
--output_dir="C:\hippojabe-generator\TunedModel" \
--instance_prompt="in the style of hippojabe" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=5e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=400 

```
### And Wait~ for it (Barney smile)

# 4. Install the CUDA Version of PyTorch
pip install torch torchvision torchaudioindex-url https://download.pytorch.org/whl/cu118

# 5. Generate images in the style of hippojabe
#### Modify the prompt on line 6 to generate different images
```
python .\GenerateImg.py
```

# Reference 
- https://huggingface.co/runwayml/stable-diffusion-v1-5?
- https://huggingface.co/docs/diffusers/v0.11.0/en/training/dreambooth
- https://blog.csdn.net/weixin_47748259/article/details/136031863
- https://medium.com/thedeephub/i-cloned-my-cousins-drawing-style-223a1fd4b093