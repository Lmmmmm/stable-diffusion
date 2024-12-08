import torch
from diffusers import StableDiffusionPipeline
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'generated_images')

def generate_kanji(prompt, checkpoint_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = StableDiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # 生成图像
    image = pipeline(prompt).images[0]
    
    # 保存图像
    save_path = os.path.join(OUTPUT_DIR, f"{prompt.replace(' ', '_')}.png")
    image.save(save_path)
    print(f"Saved image to {save_path}")

def main():
    # 获取最新的checkpoint
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('epoch_')]
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, sorted(checkpoints)[-1])
    
    while True:
        prompt = input("Enter prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
        generate_kanji(prompt, latest_checkpoint)

if __name__ == "__main__":
    main()
