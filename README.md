# About the XML Converter and Viewer

# Dependencies
- matplotlib
- opencv-python
- scipy
- shapely
# File Types
- `<image>.jpg`
    - Input image file
    - `<image>`: Image name used for all files below
- `<image>_annotate.json`
    - Input json file with text box annotations
- `<image>_tagged.json`
    - Input json file with text box annotations and tagged regions
- `<image>.json`
    - Converter output json file for viewer
- `<image>.xml`
    - Converter output json file for viewer

# Running the Converter
Converts Input JSON to XML & JSON used for the Viewer.

`python json_to_xml.py --input_dir --suffix`

`--input dir`: input directory of images and json files

`--suffix`: suffix of json file (ex. _annotate or _tagged)

**For Window Users: `encoding="utf-8"` must be added on line 1058.

# Running the Viewer
Shows annotations and tags on input image.

`python viewer.py --input_dir --suffix`

`--input dir`: input directory of images and json files

`--suffix`: suffix of json file (ex. _annotate or _tagged)
