import xml.etree.ElementTree as ET
import os
import re
import cairosvg
from PIL import Image
import random

DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
SVG_DIR = os.path.join(DATA_DIR, 'kanji_svg')
JPG_DIR = os.path.join(DATA_DIR, 'kanji_jpg')

def enhance_contrast(image):
    """Enhance image contrast and convert to binary image"""
    return image.point(lambda x: 0 if x < 128 else 255)

def validate_stroke_color(image):
    """Verify if strokes are pure black"""
    pixels = image.getdata()
    return all(p in (0, 255) for p in pixels)

def augment_image(image):
    """Image augmentation: add slight rotation and scaling"""
    angle = random.uniform(-5, 5)
    scale = random.uniform(0.95, 1.05)
    return image.rotate(angle).resize(
        (int(128*scale), int(128*scale))
    ).resize((128, 128))

def process_svg_files():
    """Process SVG files and generate PNG"""
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
    """Convert SVG to JPG images"""
    if not os.path.exists(JPG_DIR):
        os.makedirs(JPG_DIR)

    for svg_file in os.listdir(SVG_DIR):
        if svg_file.endswith('.svg'):
            svg_file_path = os.path.join(SVG_DIR, svg_file)
            jpg_file_path = os.path.join(JPG_DIR, svg_file.replace('.svg', '.jpg'))

            # Convert SVG to JPG
            cairosvg.svg2png(
                url=svg_file_path,
                write_to=jpg_file_path,
                background_color="white"
            )

            # Image processing
            with Image.open(jpg_file_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize((128, 128), Image.LANCZOS)
                img = enhance_contrast(img)  # Enhance contrast

                if validate_stroke_color(img): # Validate stroke color
                    img = augment_image(img)  # Apply image augmentation
                    img.save(jpg_file_path, 'JPEG', quality=100)
                else:
                    print(f"Warning: Invalid stroke color in {svg_file}")

def process_kanjivg():
    """Get character to filename mapping from kanjivg.xml"""
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
    """Create metadata.jsonl file and clean up irrelevant images"""
    # Get character to filename mapping
    lit2name = process_kanjivg()

    # Parse kanjidic2.xml
    kanjidic_root = ET.parse(os.path.join(RAW_DIR, 'kanjidic2.xml')).getroot()
    metadata_file_path = os.path.join(JPG_DIR, 'metadata.jsonl')

    # Create metadata.jsonl
    with open(metadata_file_path, 'w', encoding='utf-8') as metadata:
        for character in kanjidic_root.findall(".//character"):
            literal = character.find("literal").text
            meanings = []
            for meaning in character.findall(".//reading_meaning/rmgroup/meaning"):
                if 'm_lang' not in meaning.attrib:  # Keep only English meanings
                    meanings.append(meaning.text)
            if meanings and literal in lit2name:
                concat_meanings = ", ".join(meanings)
                metadata.write(f'{{"file_name": "{lit2name[literal]}.jpg", "text": "a Kanji meaning {concat_meanings}"}}\n')

    # Clean up images not in metadata
    with open(metadata_file_path, 'r') as metadata:
        metadata_content = metadata.read()
        for jpg_file in os.listdir(JPG_DIR):
            if jpg_file.endswith('.jpg') and jpg_file not in metadata_content:
                os.remove(os.path.join(JPG_DIR, jpg_file))

def main():
    try:
        # Ensure all necessary directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)

        print("Starting SVG file processing...")
        process_svg_files()

        print("Converting image formats...")
        convert_to_images()

        print("Generating metadata file...")
        create_metadata()

        print("Data processing completed!")

        # Clean up temporary SVG files
        if os.path.exists(SVG_DIR):
            import shutil
            shutil.rmtree(SVG_DIR)

    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()