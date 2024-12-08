import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image

# 定义路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'generated_images')

def generate_kanji(
    prompt,
    checkpoint_path,
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50
):
    """
    生成汉字图像
    Args:
        prompt: 英文描述
        checkpoint_path: 模型检查点路径
        num_images: 生成图像数量
        guidance_scale: 生成引导程度
        num_inference_steps: 推理步数
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = StableDiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # 生成图像
    images = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images

    # 保存图像
    for idx, image in enumerate(images):
        save_path = os.path.join(OUTPUT_DIR, f"{prompt.replace(' ', '_')}_{idx}.png")
        image.save(save_path)
        print(f"Saved image to {save_path}")

def main():
    # 获取最新的检查点
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('epoch_')]
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, sorted(checkpoints)[-1])

    print(f"Using checkpoint: {latest_checkpoint}")

    # 测试用的提示词列表
    test_prompts = [
        "a Kanji meaning mountain",
        "a Kanji meaning water",
        "a Kanji meaning fire",
        # 添加你想测试的其他提示词
    ]

    # 为每个提示词生成图像
    for prompt in test_prompts:
        print(f"\nGenerating images for prompt: {prompt}")
        generate_kanji(
            prompt=prompt,
            checkpoint_path=latest_checkpoint,
            num_images=3  # 每个提示词生成3张图像
        )

if __name__ == "__main__":
    main()