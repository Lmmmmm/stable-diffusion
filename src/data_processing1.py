# from diffusers import StableDiffusionPipeline
# import torch
# import numpy as np
# from PIL import Image

# def prepare_dataset(image_size=256):
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#     ])
#     return dataset

import xml.etree.ElementTree as ET
import os
import cairosvg
from PIL import Image
import io
import numpy as np

def parse_kanjidic2(xml_path):
    """解析KANJIDIC2 XML文件获取汉字和英文释义"""
    try:
        # 读取XML文件内容
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # 移除DOCTYPE声明
        doctype_end = xml_content.find(']>')
        if doctype_end != -1:
            xml_content = xml_content[doctype_end + 2:]

        # 解析XML
        root = ET.fromstring(xml_content)
        kanji_data = {}

        # 遍历所有character元素
        for character in root.findall('.//character'):
            literal = character.find('literal')
            if literal is not None:
                literal = literal.text
                meanings = []

                # 查找reading_meaning元素下的meaning元素
                for rm in character.findall('.//reading_meaning/rmgroup/meaning'):
                    # 只获取英文释义（没有语言属性或语言属性为en的）
                    if 'm_lang' not in rm.attrib or rm.attrib['m_lang'] == 'en':
                        if rm.text:
                            meanings.append(rm.text)

                if meanings:
                    kanji_data[literal] = ' '.join(meanings)

        return kanji_data

    except Exception as e:
        print(f"Error parsing KANJIDIC2: {str(e)}")
        raise

# def parse_kanjivg(xml_path):
#     """解析KanjiVG XML文件"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # 获取正确的命名空间
#     ns = {'kvg': 'http://kanjivg.tagaini.net'}

#     # 查找所有kanji元素，包括其路径数据
#     kanji_elements = []
#     for kanji in root.findall('.//kvg:kanji', namespaces=ns):
#         # 获取所有路径数据
#         paths = kanji.findall('.//kvg:path', namespaces=ns)
#         if paths:
#             kanji_elements.append(kanji)

#     return kanji_elements
def parse_kanjivg(xml_path):
    """解析KanjiVG XML文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取命名空间
        ns = {'kvg': 'http://kanjivg.tagaini.net'}

        # 直接查找所有kanji元素
        kanji_elements = root.findall('kvg:kanji', ns)

        if not kanji_elements:
            # 如果找不到，尝试其他查找方式
            kanji_elements = root.findall('.//kanji')

        if not kanji_elements:
            # 从搜索结果中可以看到实际的XML结构
            kanji_elements = root.findall('.//{http://kanjivg.tagaini.net}kanji')

        # 过滤出有效的kanji元素
        valid_elements = []
        for kanji in kanji_elements:
            # 检查是否包含path元素
            paths = kanji.findall('.//kvg:path', ns) or kanji.findall('.//path')
            if paths:
                g_element = kanji.find('kvg:g', ns) or kanji.find('g')
                if g_element is not None:
                    valid_elements.append(kanji)

        print(f"Found {len(valid_elements)} valid kanji elements")
        return valid_elements

    except Exception as e:
        print(f"Error parsing KanjiVG: {str(e)}")
        raise

def process_svg_to_image(kanji_element, size=256):
    """将KanjiVG元素转换为SVG图像"""
    try:
        # 获取所有路径数据
        paths = kanji_element.findall('.//{http://kanjivg.tagaini.net}path')
        if not paths:
            return None

        # 构建SVG内容
        path_data = '\n'.join([ET.tostring(path, encoding='unicode') for path in paths])

        svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 109 109">
            <g transform="scale(1,-1) translate(0,-900)" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round">
                {path_data}
            </g>
        </svg>'''

        # 转换为PNG
        png_data = cairosvg.svg2png(
            bytestring=svg_template.encode('utf-8'),
            output_width=size,
            output_height=size
        )

        # 转换为黑白图像
        img = Image.open(io.BytesIO(png_data)).convert('L')
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        return np.array(img)
    except Exception as e:
        print(f"Error processing SVG: {e}")
        return None

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Processing KANJIDIC2...")
    kanji_meanings = parse_kanjidic2("data/raw/kanjidic2.xml")
    print(f"Found {len(kanji_meanings)} kanji definitions")

    print("Processing KanjiVG...")
    kanji_elements = parse_kanjivg("data/raw/kanjivg.xml")
    print(f"Found {len(kanji_elements)} kanji in SVG file")

    dataset = []
    total = len(kanji_elements)

    for i, kanji in enumerate(kanji_elements):
        try:
            # 获取字符ID
            kanji_id = kanji.get('id', '').split(':')[-1]
            if kanji_id and kanji_id in kanji_meanings:
                # 处理SVG图像
                image = process_svg_to_image(kanji)
                if image is not None:
                    dataset.append({
                        'kanji': kanji_id,
                        'meaning': kanji_meanings[kanji_id],
                        'image': image
                    })

                    if len(dataset) % 100 == 0:
                        print(f"Successfully processed {len(dataset)}/{i+1} kanji")

        except Exception as e:
            print(f"Error processing kanji {kanji_id}: {e}")
            continue

    if not dataset:
        raise ValueError("No data was processed successfully")

    print(f"Saving {len(dataset)} processed entries...")
    np.save("data/processed/kanji_dataset.npy", dataset)
    print("Done!")


if __name__ == "__main__":
    main()

# def process_svg_to_image(kanji_element, size=256):
#     """将KanjiVG元素转换为SVG图像"""
#     try:
#         # 提取g元素
#         g_element = kanji_element.find('.//{http://kanjivg.tagaini.net}g')
#         if g_element is None:
#             g_element = kanji_element.find('.//g')

#         if g_element is None:
#             return None

#         # 创建完整的SVG文档
#         svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
#         <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 109 109">
#             <g transform="scale(0.9,0.9) translate(10,10)" fill="none" stroke="#000000" stroke-width="3">
#                 {ET.tostring(g_element, encoding='unicode')}
#             </g>
#         </svg>'''

#         # 转换为PNG
#         png_data = cairosvg.svg2png(
#             bytestring=svg_template.encode('utf-8'),
#             output_width=size,
#             output_height=size
#         )

#         # 转换为黑白图像
#         img = Image.open(io.BytesIO(png_data)).convert('L')
#         img = img.point(lambda x: 0 if x < 128 else 255, '1')
#         return np.array(img)
#     except Exception as e:
#         print(f"Error processing SVG: {e}")
#         return None

# def process_svg_to_image(kanji_element, size=256):
#     """将KanjiVG元素转换为SVG图像"""
#     try:
#         # 创建SVG头部
#         svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
#         <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 109 109">
#             <g transform="scale(1,-1) translate(0,-900)">
#                 {ET.tostring(kanji_element, encoding='unicode')}
#             </g>
#         </svg>'''

#         # 转换为PNG
#         png_data = cairosvg.svg2png(
#             bytestring=svg_template.encode('utf-8'),
#             output_width=size,
#             output_height=size
#         )

#         # 转换为黑白图像
#         img = Image.open(io.BytesIO(png_data)).convert('L')
#         img = img.point(lambda x: 0 if x < 128 else 255, '1')
#         return np.array(img)
#     except Exception as e:
#         print(f"Error processing SVG: {e}")
#         return None

# def main():
#     os.makedirs("data/processed", exist_ok=True)

#     print("Processing KANJIDIC2...")
#     kanji_meanings = parse_kanjidic2("data/raw/kanjidic2.xml")
#     print(f"Found {len(kanji_meanings)} kanji definitions")

#     print("Processing KanjiVG...")
#     kanji_elements = parse_kanjivg("data/raw/kanjivg.xml")
#     print(f"Found {len(kanji_elements)} kanji in SVG file")

#     dataset = []
#     for kanji in kanji_elements:
#         try:
#             # 获取字符ID
#             kanji_id = kanji.get('id', '').split(':')[-1]
#             if kanji_id and kanji_id in kanji_meanings:
#                 # 处理SVG图像
#                 image = process_svg_to_image(kanji)
#                 if image is not None:
#                     dataset.append({
#                         'kanji': kanji_id,
#                         'meaning': kanji_meanings[kanji_id],
#                         'image': image
#                     })

#                     if len(dataset) % 100 == 0:
#                         print(f"Processed {len(dataset)} kanji")
#         except Exception as e:
#             print(f"Error processing kanji {kanji_id}: {e}")
#             continue

#     if not dataset:
#         raise ValueError("No data was processed successfully")

#     print(f"Saving {len(dataset)} processed entries...")
#     np.save("data/processed/kanji_dataset.npy", dataset)
#     print("Done!")


# def parse_kanjivg(xml_path):
#     """解析KanjiVG XML文件"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # 从XML内容中提取命名空间
#     ns = {'kvg': 'http://kanjivg.tagaini.net'}

#     # 使用命名空间查找所有kanji元素
#     kanji_elements = root.findall('.//kvg:kanji', namespaces=ns)

#     if not kanji_elements:
#         # 如果使用命名空间找不到，尝试直接查找
#         kanji_elements = root.findall('.//kanji')

#     if not kanji_elements:
#         raise ValueError(f"在{xml_path}中未找到kanji元素")

#     print(f"Found {len(kanji_elements)} kanji elements")
#     return kanji_elements

# def process_svg_to_image(svg_content, size=256):
#     """将SVG转换为像素图像"""
#     try:
#         # 转换SVG为PNG
#         png_data = cairosvg.svg2png(bytestring=svg_content,
#                                   output_width=size,
#                                   output_height=size)

#         # 转换为黑白图像
#         img = Image.open(io.BytesIO(png_data)).convert('L')
#         # 二值化处理
#         img = img.point(lambda x: 0 if x < 128 else 255, '1')
#         return np.array(img)
#     except Exception as e:
#         print(f"Error processing SVG: {e}")
#         return None

# def main():
#     # 创建输出目录
#     os.makedirs("data/processed", exist_ok=True)

#     print("Processing KANJIDIC2...")
#     try:
#         kanji_meanings = parse_kanjidic2("data/raw/kanjidic2.xml")
#         print(f"Found {len(kanji_meanings)} kanji definitions")
#     except Exception as e:
#         print(f"Error processing KANJIDIC2: {str(e)}")
#         return

#     print("Processing KanjiVG...")
#     try:
#         kanji_elements = parse_kanjivg("data/raw/kanjivg.xml")
#         print(f"Found {len(kanji_elements)} kanji in SVG file")
#     except Exception as e:
#         print(f"Error processing KanjiVG: {str(e)}")
#         return

#     dataset = []
#     for kanji in kanji_elements:
#         try:
#             # 获取id属性
#             kanji_id = kanji.get('{http://kanjivg.tagaini.net}id', kanji.get('id'))
#             if kanji_id:
#                 literal = kanji_id.split(':')[-1]

#                 if literal in kanji_meanings:
#                     # 将XML元素转换为字符串
#                     svg_content = ET.tostring(kanji, encoding='unicode')
#                     image = process_svg_to_image(svg_content)

#                     if image is not None:
#                         dataset.append({
#                             'kanji': literal,
#                             'meaning': kanji_meanings[literal],
#                             'image': image
#                         })

#                 if len(dataset) % 100 == 0:
#                     print(f"Processed {len(dataset)} kanji")

#         except Exception as e:
#             print(f"Error processing kanji: {str(e)}")
#             continue
#     if len(dataset) == 0:
#         raise ValueError("No data was processed successfully")

#     print(f"Saving {len(dataset)} processed entries...")
#     np.save("data/processed/kanji_dataset.npy", dataset)
#     print("Done!")



# def parse_kanjidic2(xml_path):
#     """解析KANJIDIC2 XML文件获取汉字和英文释义"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     kanji_data = {}

#     for character in root.findall('.//character'):
#         literal = character.find('literal').text
#         meanings = character.findall(".//meaning[@m_lang='en']")
#         if meanings:
#             # 合并所有英文释义
#             meaning = ' '.join([m.text for m in meanings])
#             kanji_data[literal] = meaning

#     return kanji_data
# def parse_kanjidic2(xml_path):
#     """解析KANJIDIC2 XML文件获取汉字和英文释义"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     kanji_data = {}

#     # 注意：kanjidic2使用不同的XML结构
#     for character in root.findall('.//character'):
#         literal = character.find('literal').text
#         meanings = character.findall(".//meaning[@m_lang='en' or not(@m_lang)]")
#         if meanings:
#             meaning = ' '.join([m.text for m in meanings])
#             kanji_data[literal] = meaning

#     return kanji_data

# def parse_kanjivg(xml_path):
#     """解析KanjiVG XML文件"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # 处理XML命名空间
#     ns = {'kvg': 'http://kanjivg.tagaini.net'}
#     return root.findall('.//kvg:kanji', namespaces=ns)

# def parse_kanjivg(xml_path):
#     """解析KanjiVG XML文件"""
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # 从XML内容中提取命名空间
#     ns = {'kvg': root.tag.split('}')[0].strip('{')}
#     kanji_elements = root.findall('.//kvg:kanji', namespaces={'kvg': ns})

#     if not kanji_elements:
#         # 尝试不使用命名空间
#         kanji_elements = root.findall('.//kanji')

#     return kanji_elements


        # dataset = []
        # total_kanji = len(root.findall('.//{http://kanjivg.tagaini.net}kanji'))
        # print(f"Found {total_kanji} kanji in SVG file")

        # for i, kanji in enumerate(root.findall('.//{http://kanjivg.tagaini.net}kanji')):
        #     try:
        #         literal = kanji.get('id').split(':')[-1]
        #         if literal in kanji_meanings:
        #             # 获取SVG内容
        #             svg_content = ET.tostring(kanji)

        #             # 转换为图像
        #             image = process_svg_to_image(svg_content)
        #             if image is not None:
        #                 dataset.append({
        #                     'kanji': literal,
        #                     'meaning': kanji_meanings[literal],
        #                     'image': image
        #                 })

        #             # 打印进度
        #             if (i + 1) % 100 == 0:
        #                 print(f"Processed {i + 1}/{total_kanji} kanji")

        #     except Exception as e:
        #         print(f"Error processing kanji {literal}: {str(e)}")
        #         continue

        # if len(dataset) == 0:
        #     raise ValueError("No data was processed successfully")

        # # 保存处理后的数据
        # print(f"Saving {len(dataset)} processed entries...")
        # np.save("data/processed/kanji_dataset.npy", dataset)
        # print("Done!")

    # except Exception as e:
    #     print(f"Error in main process: {str(e)}")
    #     raise

# if __name__ == "__main__":
#     main()