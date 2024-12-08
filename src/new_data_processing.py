import xml.etree.ElementTree as ET
import os
import re
import requests
import io
import cairosvg
import shutil
from PIL import Image

"""
Generates SVG files from kanjivg.xml in folder kanji_svg
"""

tree = ET.parse("data/raw/kanjivg.xml")
root = tree.getroot()
svg_folder = "kanji_svg"

if not os.path.exists(svg_folder):
    os.makedirs(svg_folder)

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

        svg_file_path = os.path.join(svg_folder, f"{kanji_id}.svg").replace("kvg:kanji_", "")
        with open(svg_file_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_content)

"""
Convert SVG files to PNG using cairosvg and then to JPG using PIL
"""

jpg_folder = 'kanji_jpg'
png_folder = 'kanji_png'

if not os.path.exists(jpg_folder):
    os.makedirs(jpg_folder)
if not os.path.exists(png_folder):
    os.makedirs(png_folder)

for svg_file in os.listdir(svg_folder):
    if svg_file.endswith('.svg'):
        svg_file_path = os.path.join(svg_folder, svg_file)
        png_file_path = svg_file_path.replace('svg', 'png')
        jpg_file_path = os.path.join(jpg_folder, svg_file.replace('.svg', '.jpg'))

        # # 转换SVG到PNG
        # cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)
        # 设置白色背景进行SVG转换
        cairosvg.svg2png(
            url=svg_file_path,
            write_to=jpg_file_path,
            background_color="white"  # 添加白色背景
        )


        # 转换为128x128黑白图像
        with Image.open(png_file_path) as img:
            # 转换为灰度图
            img = img.convert('L')
            # 调整大小为128x128
            img = img.resize((128, 128), Image.LANCZOS)
            # 创建白色背景
            img.save(jpg_file_path, 'JPEG', quality=85, optimize=True)
            # background = Image.new('L', img.size, 'white')
            # background.paste(img, (0, 0), img)
            # background.save(jpg_file_path, 'JPEG')

# for svg_file in os.listdir(svg_folder):
#     if svg_file.endswith('.svg'):
#         svg_file_path = os.path.join(svg_folder, svg_file)
#         png_file_path = svg_file_path.replace('svg', 'png')
#         jpg_file_path = os.path.join(jpg_folder, svg_file.replace('.svg', '.jpg'))

#         cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)

#         with Image.open(png_file_path) as img:
#             with Image.new('RGB', img.size, 'WHITE') as background:
#                 background.paste(img, (0, 0), img)
#                 background.save(jpg_file_path, 'JPEG')

"""
Map the kanji filename to the kanji literal using kanjivg.xml
"""

kvg_element_pattern = re.compile(r'kvg:element="([^"]+)"')
lit2name = {}
is_above_kanji = False

with open('data/raw/kanjivg.xml', 'r', encoding='utf-8') as kanjivg:
    for line in kanjivg:
        if '<kanji' in line:
            is_above_kanji = True
        if is_above_kanji:
            kanji_id = re.search(r'id="([^"]+)"', line)
            lit = kvg_element_pattern.search(line)
            if lit:
                lit2name[lit.group(1)] = kanji_id.group(1).replace('kvg:', '')
                is_above_kanji = False

"""
Map the kanji filename to the English kanji meaning using kanjidic2.xml
"""

kanjidic_root = ET.parse('data/raw/kanjidic2.xml').getroot()
metadata_file_path = os.path.join(jpg_folder, 'metadata.jsonl')

with open(metadata_file_path, 'w', encoding='utf-8') as metadata:
    for character in kanjidic_root.findall(".//character"):
        literal = character.find("literal").text
        meanings = []
        for meaning in character.findall(".//reading_meaning/rmgroup/meaning"):
            if 'm_lang' not in meaning.attrib:  # Only English meanings
                meanings.append(meaning.text)
        if meanings and literal in lit2name:
            concat_meanings = ", ".join(meanings)
            metadata.write(f'{{"file_name": "{lit2name[literal]}.jpg", "text": "a Kanji meaning {concat_meanings}"}}\n')

"""
Remove images in the JPG folder that are not in metadata.jsonl
"""

with open(metadata_file_path, 'r') as metadata:
    metadata_content = metadata.read()
    for jpg_file in os.listdir(jpg_folder):
        if jpg_file.endswith('.jpg') and jpg_file not in metadata_content:
            os.remove(os.path.join(jpg_folder, jpg_file))


# import xml.etree.ElementTree as ET
# import os
# import re
# import requests
# import io
# import cairosvg
# import shutil
# from PIL import Image


# """
# Generates SVG files from kanjivg.xml in folder kanji_svg
# """

# tree = ET.parse("data/raw/kanjivg.xml")

# root = tree.getroot()

# svg_folder = "kanji_svg"

# if not os.path.exists(svg_folder):
#     os.makedirs(svg_folder)

# kanji_header = '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'
# kanji_style = 'style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'

# # Assuming each kanji element has an 'id' attribute to use as filename
# for kanji in root:
#     kanji_id = kanji.attrib.get("id")
#     if kanji_id:
#         svg_content = f"{kanji_header}\n"
#         for g in kanji.findall(".//g"):
#             g_str = ET.tostring(g, encoding="utf-8", method="xml").decode("utf-8")
#             svg_content += f"<g {kanji_style}{g_str}</g>\n"

#         svg_content += "</svg>"

#         svg_file_path = os.path.join(svg_folder, f"{kanji_id}.svg").replace(
#             "kvg:kanji_", ""
#         )

#         with open(svg_file_path, "w", encoding="utf-8") as svg_file:
#             svg_file.write(svg_content)

# """
# Convert SVG files to PNG using cairosvg and then to JPG using PIL
# """

# jpg_folder = 'kanji_jpg'
# png_folder = 'kanji_png'

# if not os.path.exists(jpg_folder):
#     os.makedirs(jpg_folder)
# if not os.path.exists(png_folder):
#     os.makedirs(png_folder)

# for svg_file in os.listdir(svg_folder):
#     if svg_file.endswith('.svg'):
#         svg_file_path = os.path.join(svg_folder, svg_file)
#         png_file_path = svg_file_path.replace('svg', 'png')
#         jpg_file_path = os.path.join(jpg_folder, svg_file.replace('.svg', '.jpg'))

#         cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)

#         # Convert PNG to JPG with a white background
#         with Image.open(png_file_path) as img:
#             with Image.new('RGB', img.size, 'WHITE') as background:
#                 background.paste(img, (0, 0), img)
#                 background.save(jpg_file_path, 'JPEG')

# """
# Map the kanji filename to the kanji literal using kanjivg.xml
# """

# kvg_element_pattern = re.compile(r'kvg:element="([^"]+)"')
# lit2name = {}
# is_above_kanji = False

# with open('data/raw/kanjivg.xml', 'r', encoding='utf-8') as kanjivg:
#     for line in kanjivg:
#         if '<kanji' in line:
#             is_above_kanji = True
#         if is_above_kanji:
#             kanji_id = re.search(r'id="([^"]+)"', line)
#             lit = kvg_element_pattern.search(line)
#             if lit:
#                 lit2name[lit.group(1)] = kanji_id.group(1).replace('kvg:', '')
#                 is_above_kanji = False

# """
# Map the kanji filename to the English kanji meaning using kanjidic2.xml and write them in metadata.jsonl
# """

# root = ET.parse('data/raw/kanjivg.xml').getroot()

# metadata_file_path = os.path.join(jpg_folder, 'metadata.jsonl')

# with open(metadata_file_path, 'w') as metadata:
#     for character in root.findall("character"):
#         lit = character.find("literal").text
#         meanings = []
#         for meaning in character.findall(".//reading_meaning/rmgroup/meaning"):
#             # Only English meanings, remove for all languages
#             if 'r_type' not in meaning.attrib and 'm_lang' not in meaning.attrib:
#                 meanings.append(meaning.text)
#         concat_meanings = ", ".join(meanings)
#         if lit in lit2name:
#             metadata.write(f'{{"file_name": "{lit2name[lit]}.jpg", "text": "a Kanji meaning {concat_meanings}"}}\n')

# """
# Remove images in the JPG folder that are not in metadata.jsonl
# """
# for jpg_file in os.listdir(jpg_folder):
#     if jpg_file.endswith('.jpg'):
#         with open(metadata_file_path, 'r') as metadata:
#             if jpg_file not in metadata.read():
#                 os.remove(os.path.join(jpg_folder, jpg_file))