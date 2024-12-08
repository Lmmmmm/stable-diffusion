## First, download the dataset(use gzip or similar CLI tool)

# We'll use the XML package in PSL. This is for the bilingual dictionary (This needs to be updated)

import xml.etree.ElementTree as ET
tree = ET.parse('kanjidic2.xml')
root = tree.getroot()

kanji_list = []
mn_list = []

for ch in root.iter('character'):  # It seems that the 0 index element is just some metadata, all entries apart from that are Kanji Characters.
    n_mn_list = []
    for lt in ch.iter('literal'):
      kanji_list.append(lt)

    for mn in ch.iter('meaning'):
      n_mn_list.append(mn.text)

    mn_list.append(n_mn_list)

assert len(kanji_list) == len(mn_list)

## This is for generating the PNG images from the SVG.

import cairosvg
import svgpathtools
import svgwrite
import xml.etree.ElementTree as ET

tree = ET.parse('kanjivg.xml')
root = tree.getroot()

paths = []
nested_paths = []
ids = []

for graphics in root.iter("g"):  # Find all path elements
    nested_paths = []
    for path in graphics.iter('path'):

      nested_paths.append(path.get("d"))

    # This will be very useful when we draw from this.
    if len(nested_paths) == 1:
      paths.append(nested_paths[0])
    else:
      paths.append(nested_paths)

    ids.append(graphics.get("id"))

assert len(paths) == len(ids)
print(len(paths))

## When there are more than one paths, this is to be done :
import cairosvg

def svg_path_to_png(svg_path, output_png_path):

    if type(svg_path) is list:
      svg_content = '<svg xmlns="http://www.w3.org/2000/svg">{}</svg>'.format(
        "".join('<path d="{}"/>'.format(path) for path in svg_path))

    elif type(svg_path) is str:
      svg_content = f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{svg_path}"/></svg>'

    cairosvg.svg2png(bytestring=svg_content.encode(), write_to=output_png_path, output_width=128, output_height=128)


for i in range(len(paths)):
  svg_path = paths[i]
  output_png_path = f"/content/kanji_png/{ids[i]}.png"
  svg_path_to_png(svg_path, output_png_path)


## Then, make it into ZIP, and download.