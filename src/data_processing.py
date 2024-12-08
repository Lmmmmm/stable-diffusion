import xml.etree.ElementTree as ET
import os
import re
import cairosvg
from PIL import Image
import random

# 定义数据目录
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
SVG_DIR = os.path.join(DATA_DIR, 'kanji_svg')
JPG_DIR = os.path.join(DATA_DIR, 'kanji_jpg')

def enhance_contrast(image):
    """提升图像对比度，将图像转换为二值图像"""
    return image.point(lambda x: 0 if x < 128 else 255)

def validate_stroke_color(image):
    """验证笔画是否为纯黑色"""
    pixels = image.getdata()
    return all(p in (0, 255) for p in pixels)

def augment_image(image):
    """图像增强：添加轻微旋转和缩放"""
    angle = random.uniform(-5, 5)
    scale = random.uniform(0.95, 1.05)
    return image.rotate(angle).resize(
        (int(128*scale), int(128*scale))
    ).resize((128, 128))

def process_svg_files():
    """处理SVG文件并生成PNG"""
    tree = ET.parse(os.path.join(RAW_DIR, 'kanjivg.xml'))
    root = tree.getroot()

    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)

    kanji_header = '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'
    kanji_style = 'style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'

    for kanji in root:
        kanji_id = kanji.attrib.get("id")
        if kanji_id:
            svg_content = f"{kanji_header}\n"
            for g in kanji.findall(".//g"):
                g_str = ET.tostring(g, encoding="utf-8", method="xml").decode("utf-8")
                svg_content += f"<g {kanji_style}{g_str}</g>\n"
            svg_content += "</svg>"

            svg_file_path = os.path.join(SVG_DIR, f"{kanji_id}.svg").replace("kvg:kanji_", "")
            with open(svg_file_path, "w", encoding="utf-8") as svg_file:
                svg_file.write(svg_content)

def convert_to_images():
    """将SVG转换为JPG图像"""
    if not os.path.exists(JPG_DIR):
        os.makedirs(JPG_DIR)

    for svg_file in os.listdir(SVG_DIR):
        if svg_file.endswith('.svg'):
            svg_file_path = os.path.join(SVG_DIR, svg_file)
            jpg_file_path = os.path.join(JPG_DIR, svg_file.replace('.svg', '.jpg'))

            # 转换SVG到JPG
            cairosvg.svg2png(
                url=svg_file_path,
                write_to=jpg_file_path,
                background_color="white"
            )

            # 图像处理
            with Image.open(jpg_file_path) as img:
                img = img.convert('L')  # 转换为灰度图
                img = img.resize((128, 128), Image.LANCZOS)
                img = enhance_contrast(img)  # 增强对比度

                if validate_stroke_color(img):  # 验证笔画颜色
                    img = augment_image(img)  # 应用图像增强
                    img.save(jpg_file_path, 'JPEG', quality=100)
                else:
                    print(f"Warning: Invalid stroke color in {svg_file}")

def process_kanjivg():
    """从kanjivg.xml获取汉字到文件名的映射"""
    kvg_element_pattern = re.compile(r'kvg:element="([^"]+)"')
    lit2name = {}
    is_above_kanji = False

    with open(os.path.join(RAW_DIR, 'kanjivg.xml'), 'r', encoding='utf-8') as kanjivg:
        for line in kanjivg:
            if '<kanji' in line:
                is_above_kanji = True
            if is_above_kanji:
                kanji_id = re.search(r'id="([^"]+)"', line)
                lit = kvg_element_pattern.search(line)
                if lit:
                    lit2name[lit.group(1)] = kanji_id.group(1).replace('kvg:', '')
                    is_above_kanji = False

    return lit2name

def create_metadata():
    """创建metadata.jsonl文件并清理无关图片"""
    # 获取汉字到文件名的映射
    lit2name = process_kanjivg()

    # 解析kanjidic2.xml
    kanjidic_root = ET.parse(os.path.join(RAW_DIR, 'kanjidic2.xml')).getroot()
    metadata_file_path = os.path.join(JPG_DIR, 'metadata.jsonl')

    # 创建metadata.jsonl
    with open(metadata_file_path, 'w', encoding='utf-8') as metadata:
        for character in kanjidic_root.findall(".//character"):
            literal = character.find("literal").text
            meanings = []
            for meaning in character.findall(".//reading_meaning/rmgroup/meaning"):
                if 'm_lang' not in meaning.attrib:  # 只保留英文含义
                    meanings.append(meaning.text)
            if meanings and literal in lit2name:
                concat_meanings = ", ".join(meanings)
                metadata.write(f'{{"file_name": "{lit2name[literal]}.jpg", "text": "a Kanji meaning {concat_meanings}"}}\n')

    # 清理不在metadata中的图片
    with open(metadata_file_path, 'r') as metadata:
        metadata_content = metadata.read()
        for jpg_file in os.listdir(JPG_DIR):
            if jpg_file.endswith('.jpg') and jpg_file not in metadata_content:
                os.remove(os.path.join(JPG_DIR, jpg_file))

def main():
    """主函数"""
    try:
        # 确保所有必要的目录都存在
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)

        print("开始处理SVG文件...")
        process_svg_files()

        print("转换图像格式...")
        convert_to_images()

        print("生成metadata文件...")
        create_metadata()

        print("数据处理完成！")

        # 清理临时SVG文件
        if os.path.exists(SVG_DIR):
            import shutil
            shutil.rmtree(SVG_DIR)

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
# import xml.etree.ElementTree as ET
# import os
# import re
# import cairosvg
# from PIL import Image
# import random
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# def enhance_contrast(image):
#     return image.point(lambda x: 0 if x < 128 else 255)

# def validate_stroke_color(image):
#     pixels = image.getdata()
#     return all(p in (0, 255) for p in pixels)

# def augment_image(image):
#     angle = random.uniform(-5, 5)
#     scale = random.uniform(0.95, 1.05)
#     return image.rotate(angle).resize(
#         (int(128*scale), int(128*scale))
#     ).resize((128, 128))

# def process_svg_files():
#     tree = ET.parse("data/raw/kanjivg.xml")
#     root = tree.getroot()
#     svg_folder = "kanji_svg"
#     jpg_folder = 'kanji_jpg'

#     if not os.path.exists(svg_folder):
#         os.makedirs(svg_folder)
#     if not os.path.exists(jpg_folder):
#         os.makedirs(jpg_folder)

#     kanji_header = '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'
#     kanji_style = 'style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'

#     for kanji in root:
#         kanji_id = kanji.attrib.get("id")
#         if kanji_id:
#             svg_content = f"{kanji_header}\n"
#             for g in kanji.findall(".//g"):
#                 g_str = ET.tostring(g, encoding="utf-8", method="xml").decode("utf-8")
#                 svg_content += f"<g {kanji_style}{g_str}</g>\n"
#             svg_content += "</svg>"

#             svg_file_path = os.path.join(svg_folder, f"{kanji_id}.svg").replace("kvg:kanji_", "")
#             with open(svg_file_path, "w", encoding="utf-8") as svg_file:
#                 svg_file.write(svg_content)

# def convert_to_images():
#     svg_folder = "kanji_svg"
#     jpg_folder = 'kanji_jpg'

#     for svg_file in os.listdir(svg_folder):
#         if svg_file.endswith('.svg'):
#             svg_file_path = os.path.join(svg_folder, svg_file)
#             jpg_file_path = os.path.join(jpg_folder, svg_file.replace('.svg', '.jpg'))

#             cairosvg.svg2png(
#                 url=svg_file_path,
#                 write_to=jpg_file_path,
#                 background_color="white"
#             )

#             with Image.open(jpg_file_path) as img:
#                 img = img.convert('L')
#                 img = img.resize((128, 128), Image.LANCZOS)
#                 img = enhance_contrast(img)
#                 if validate_stroke_color(img):
#                     img = augment_image(img)
#                     img.save(jpg_file_path, 'JPEG', quality=100)

# def create_metadata():
#     kvg_element_pattern = re.compile(r'kvg:element="([^"]+)"')
#     lit2name = {}

#     with open('data/raw/kanjivg.xml', 'r', encoding='utf-8') as kanjivg:
#         for line in kanjivg:
#             if '<kanji' in line:
#                 kanji_id = re.search(r'id="([^"]+)"', line)
#                 lit = kvg_element_pattern.search(line)
#                 if lit and kanji_id:
#                     lit2name[lit.group(1)] = kanji_id.group(1).replace('kvg:', '')

#     kanjidic_root = ET.parse('data/raw/kanjidic2.xml').getroot()
#     metadata_file_path = os.path.join('kanji_jpg', 'metadata.jsonl')

#     with open(metadata_file_path, 'w', encoding='utf-8') as metadata:
#         for character in kanjidic_root.findall(".//character"):
#             literal = character.find("literal").text
#             meanings = []
#             for meaning in character.findall(".//reading_meaning/rmgroup/meaning"):
#                 if 'm_lang' not in meaning.attrib:
#                     meanings.append(meaning.text)
#             if meanings and literal in lit2name:
#                 concat_meanings = ", ".join(meanings)
#                 metadata.write(f'{{"file_name": "{lit2name[literal]}.jpg", "text": "a Kanji meaning {concat_meanings}"}}\n')

# class KanjiDataset(Dataset):
#     def __init__(self, data_dir='kanji_jpg', metadata_file='metadata.jsonl'):
#         self.data_dir = data_dir
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])

#         self.data = []
#         with open(os.path.join(data_dir, metadata_file), 'r') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image_path = os.path.join(self.data_dir, item['file_name'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         return {
#             'image': image,
#             'meaning': item['text']
#         }

# def main():
#     process_svg_files()
#     convert_to_images()
#     create_metadata()

# if __name__ == "__main__":
#     main()