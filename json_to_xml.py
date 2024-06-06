# PAGE REF: https://www.primaresearch.org/schema/PAGE/gts/pagecontent/2019-07-15/Simple%20PAGE%20XML%20Example.pdf
import sys
import points
import poly_routines as poly
import json_to_xml as xml
import xml.etree.ElementTree as ET
import xml.dom.minidom as dom
import json
from PIL import Image, ImageOps
import os
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import re
import datetime
#import alphashape
import shapely
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon, Polygon


TAG_KEY = "tags"
LINE_KEY_START = "line_"
DELETED_KEY = "deleted"
REGION_TAG_START = "Region_"
PRINTED_TAG_START = "Printed_"
THRESHOLD_PERCENT_AREA = 0.80
ARABIC = "Arabic"
ENGLISH = "English"
LATIN = "Latin"
ENG_TAG = '#en#'
ENG_END_TAG = '#end_en#'
MIN_POLY_PTS = 3
ARCHIVIST_TAG = "Archivist_Tag"
CROSSED_TAG = "Crossed_out"
NUMERIC = "Numeric"
ARCHIVE_NUMBER = "Archive_Number"
ORIENT_TOP_BOTTOM_TAG = "Orient_top_bottom"
ORIENT_BOTTOM_TOP_TAG = "Orient_bottom_top"
#IMAGE_TAG = 'Image'
# noise, separator, margin are left
REGION_TYPE_DICT = {'paragraph': {'name': 'paragraph', 'mainType': 'text', 'subType': 'paragraph'}, 
                    'floating': {'name': 'floating', 'mainType': 'text', 'subType': 'floating'},
                    'heading': {'name': 'heading', 'mainType': 'text', 'subType': 'heading'}, 
                    'logo': {'name': 'logo', 'mainType': 'graphic', 'subType': 'logo'}, 
                    'decoration': {'name': 'decoration', 'mainType': 'graphic', 'subType': 'decoration'}, 
                    'letterhead': {'name': 'letterhead', 'mainType': 'graphic', 'subType': 'letterhead'}, 
                    'pageNo': {'name': 'pageNo', 'mainType': 'text', 'subType': 'page-number'}, 
                    'signature': {'name': 'signature', 'mainType': 'text', 'subType': 'signature-mark'},
                    'graphic': {'name': 'graphic', 'mainType': 'graphic', 'subType': 'other'}, 
                    'image': {'name': 'image', 'mainType': 'image', 'subType': 'image'}, 
                    'stamp': {'name': 'stamp', 'mainType': 'graphic', 'subType': 'stamp'}, 
                    'margin': {'name': 'margin', 'mainType': 'text', 'subType': 'floating'}, 
                    'noise': {'name': 'noise', 'mainType': 'noise', 'subType': 'noise'},
                    'separator': {'name': 'separator', 'mainType': 'separator', 'subType': 'separator'}}

# These will be placed as a group for sorting
MAIN_PAGE_REGION_NAMES = ['paragraph', 'heading', 'separator', 'signature']
TARGET_SUFFIX = '_tagged'
def get_json_file(full_img_name, target_suffix=None):
    if target_suffix is None:
        target_suffix = TARGET_SUFFIX
    json_files = []
    #annotators = []
    dir, img_name = os.path.split(full_img_name)
    base_file = img_name[:-4]
    files = os.listdir(dir)
    for f in files:
        prefix = base_file + target_suffix
        
        if f.startswith(prefix) and f.endswith('json'):
            
            # Check if its a timestamp in filename
            partial_string = f[len(prefix):]
            ind1 = partial_string.rfind('.')
            ind2 = partial_string.find('.')
            #annotator = partial_string[:ind1]
            
            if (ind1 == ind2):
                json_files.append(f)
                #annotators.append(annotator)
    if len(json_files) == 0:
        return None
    if len(json_files) > 1:
        print('More than one json found...returning 0th one', json_files, annotators)
    
    
    return os.path.join(dir, json_files[0])


class LanguageRoutines:
    def __init__(self):
        

        english_lower = range(ord('a'), ord('z')+1)
        english_upper = range(ord('A'), ord('Z')+1)
        english_numbers = range(ord('0'), ord('9')+1)

        english_ord = set(english_lower).union(english_upper).union(english_numbers)
        english_numbers = {chr(c) for c in set(english_numbers)}
        self.english_alphabet = {chr(c) for c in english_ord}

        arabic_unicodes = range(ord("\u0600"), ord("\u06ff")+1)
        arabic_ord = set(arabic_unicodes)
        self.arabic_chars = {chr(c) for c in arabic_ord}


    # Add english tag if necessary. If Arabic leave the text as is
    def detect_language_and_add_tags(self, text):
        en_start = -1
        en_end = -1
        english_in_text = []
        for ind, c in enumerate(text.lower()):
            if c in self.english_alphabet:
                
                if en_start == -1:
                    #print(ind, c)
                    en_start = ind
                    en_end = ind
                else:
                    
                    en_end = ind
            elif c not in self.english_alphabet and c not in self.arabic_chars and en_start != -1:
                
                en_end = ind
            else:
                if en_start != -1:
                    english_in_text.append([en_start, en_end])
                en_start = -1
                en_end = -1
        # Check if all English
        if en_start==0 and en_end==len(text)-1:
            language = ENGLISH
        else:
            language = ARABIC
            if en_start != -1 and en_end != -1:
                english_in_text.append([en_start, en_end])

        
        new_text = ''
        text_ind = 0
        # Create a new text line with tags
        if len(english_in_text) > 0 and language != ENGLISH:
            for [start_ind, end_ind] in english_in_text:
                
                if start_ind > 0:
                    new_text = new_text + text[text_ind:start_ind-1] + ENG_TAG + text[start_ind: end_ind+1] + ENG_END_TAG
                else: 
                    new_text = ENG_TAG + text[start_ind: end_ind+1] + ENG_END_TAG
                text_ind = end_ind+1
        new_text = new_text + text[text_ind:]
        return language, new_text
    
    # Not used
    def is_english(self, text):
        text = text.strip()
        
        en_start = -1
        en_end = -1
        
        for ind, c in enumerate(text.lower()):
            if c in self.english_alphabet:
                
                if en_start == -1:
                    
                    en_start = ind
                    en_end = ind
                else:                   
                    en_end = ind
            elif c not in self.english_alphabet and c not in self.arabic_chars and en_start != -1:
                en_end = ind
            else:
                break
        # Check if all English
        if en_start==0 and en_end==len(text)-1:
            language = ENGLISH
        else:
            language = ARABIC
        
        if language == ENGLISH:
            return True
        return False
    
    def is_numeric(self, input_text):
        # Remove punctuation 
        text = re.sub(r'[^\w]', '', input_text)
        # Remove underscores
        text = text.replace('_', '')
        return text.isdigit()
    
    def is_latin(self, input_text):
        # First isolate all chars
        text = [t for t in input_text if t.isalpha() or t.isdigit() ]
        if len(text) == 0:
            latin = False
        else:
            latin = True

        # CHeck if all are latin
        for t in text:
            if t in self.arabic_chars:
                latin = False
                break
        return latin
        

    
# For saving to json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)    

class TranscriptionPage:
    def __init__(self, json=None, filename=None, imagefile=None, load_from_file=None):
        
        if load_from_file is not None:
            self.load_from_file = load_from_file
            self.load()
            
            return
        
        
        self.json_filename = filename
        if json is None and filename is not None:
            self.json = self.get_json(filename)
        else:
            self.json = json
            

        self.keys = list(self.json.keys())

        if imagefile is not None:
            self.image, self.size = self.get_image(imagefile)
            self.imagedir, self.imagefile = os.path.split(imagefile)
            
        self.detector = LanguageRoutines()    
        
        self.line_dict = dict()
        self.region_dict = dict()
        self.page_dict = dict()
        self.meta_dict = dict()
        self.setup_dictionaries()
        
        self.set_line_properties()
        self.set_region_properties()
        self.assign_lines_to_regions()
        self.get_text_region()
        self.recalc_region_coords()
        self.assign_ids_to_all()
        self.sort_all()

        self.sorted_regions = self.sort_regions()
        

        
    def load(self):
        with open(self.load_from_file) as fin:
            main_obj = json.load(fin)
        self.imagedir, self.imagefile = main_obj['imagedir'], main_obj['imagefile']    
        self.json_filename = main_obj['json_filename']
        self.line_dict = main_obj['line_dict']
        self.region_dict = main_obj['region_dict']
        self.page_dict = main_obj['page_dict']
        self.sorted_regions = main_obj['sorted_regions']
        self.json = main_obj['json']
        
        
    def save(self):
        main_obj = dict()
        main_obj['imagedir'], main_obj['imagefile'] = self.imagedir, self.imagefile 
        main_obj['json_filename'] = self.json_filename
        main_obj['line_dict'] = self.line_dict
        main_obj['region_dict'] = self.region_dict 
        main_obj['page_dict'] = self.page_dict 
        main_obj['sorted_regions'] = self.sorted_regions 
        main_obj['json'] = self.json       

        main_json = os.path.join(self.imagedir, self.imagefile[:-3]+'json')
        with open(main_json, 'w') as fout:
            json.dump(main_obj, fout, cls=NpEncoder, indent=2)
        

    # For assigning unique number to regions and lines
    # This id can be used in xml ... 
    def assign_ids_to_all(self):
        
        total = 0
        for k in self.region_dict.keys():
            if not 'id' in self.json[k]:
                self.json[k]['id'] = f'{total}'
                total += 1
                
        for k in self.line_dict.keys():
            if not 'id' in self.json[k]:
                self.json[k]['id'] = f'{total}'
                total += 1
    
    def get_json(self, filename):
        with open(filename) as fin:
            json_obj = json.load(fin)
        return json_obj

    # Set properties of text lines
    def set_line_properties(self):
        for k in self.line_keys():
            # set language
            self.get_language(k)
            # set production and styles
            self.get_production(k)
            
            
    def set_region_properties(self):
        for k,v in self.region_dict.items():
            self.json[k][ARCHIVE_NUMBER] = False
            # IF the region is floating and archivist tag is set and there is one line in it
            # And the line is numeric, then we'll set archivist number to true
            if v['type'] == 'floating' and len(v['lines']) == 1:
                text = self.get_text(k)
                if self.detector.is_numeric(text) and self.get_tag(k, ARCHIVIST_TAG):
                    self.json[region_key][ARCHIVE_NUMBER] = True
                
    
    def recalc_region_coords(self):
        #print(self.region_dict)
        #print(self.line_dict)
        for r,v in self.region_dict.items():
            if v['type']['mainType'] == 'graphic' or v['type']['mainType'] == 'image':
                # THis will have coords of graphics region
                # and its corresponding json will have text region coords
                v['coord'] = self.json[r]["coord"]
                # Do not recalc
                continue
            lines_in_region = len(v['lines'])
            #print(r, ':', lines_in_region)
            # Recalc only if multiple lines in region
            if lines_in_region > 1:
                #print('computing coords', r) 
                coords = self.get_convex_hull(v['lines'])
                self.json[r]["coord"] = coords
                self.json[r]["line_polygon"] = self.list_to_xy(coords)
            elif lines_in_region == 1:
                line_key = v['lines'][0]
                self.json[r]["coord"] = self.json[line_key]["coord"]                
                line_polygon = self.list_to_xy(self.json[r]["coord"])
                self.json[r]["line_polygon"] = line_polygon

        
    def get_annotator(self):
        
        index1 = self.json_filename.find('_annotate_') + 10
        return self.json_filename[index1:-5]
        
    def get_image(self, imagefile):
        try:
            image = Image.open(imagefile)
            image = ImageOps.exif_transpose(image)
            size = image.size
        except Exception as e:
            # If image not opened
            img = cv2.imread(imagefile)
            # save the image
            cv2.imwrite(imagefile, img)
            image = Image.open(imagefile)
            image = ImageOps.exif_transpose(image)
            size = image.size
            
        return image, size
    
    def get_meta_items_dict(self):
        items = dict()
        values = ['writer', 'comment', 'taggingBy', 'transcription_QA']
        keys = ['author', 'other1', 'other2', 'other3']
        
        for k,v in zip(keys, values):
            value = self.get_value(v)
            
            if len(value) > 0:
                if k.startswith('other'):
                    value = f'{v}:{value}'
                items[k] = value
        return items
        
    # set up line_dict and region_dict
    def setup_dictionaries(self):

        date_time = datetime.datetime.utcnow()
        date_str = f'{date_time.year}-{date_time.month:02d}-{date_time.day:02d}T{date_time.hour:02d}:{date_time.minute:02d}:{date_time.second:02d}'
        
        line_keys = self.get_line_keys(self.keys)
        self.line_dict = {k:{'assigned': False, 'region': None} for k in line_keys}
        
        region_keys, region_types = self.get_regions()
        self.region_dict = {k:{'lines': [], 'type': t} for k, t in zip(region_keys, region_types)}

        self.page_dict = {'imageFilename': self.imagefile, 'imageWidth': str(self.size[0]), 
                          'imageHeight': str(self.size[1])}
        # TODO: Add the created/modified date from json
        self.meta_dict = {'Creator': self.get_annotator(), 'Created':date_str, 'LastChange':date_str}
        self.meta_items_dict = self.get_meta_items_dict()
    
    def line_keys(self):
        return self.line_dict.keys()
    
    def region_keys(self):
        return self.region_dict.keys()
        
    def is_valid_line(self, line_key):
        valid = False
        if line_key.startswith(LINE_KEY_START) and self.get_value(line_key, DELETED_KEY) != '1'\
            and len(self.get_value(line_key, "coord", [])) >= MIN_POLY_PTS*2:
            valid = True
        return valid
        
    def get_line_keys(self, key_list):
        line_keys = [k for k in key_list if self.is_valid_line(k)]
        return line_keys



    def get_value(self, key, sub_key=None, default=""):
        if sub_key is None:
            if key in self.json.keys():
                return self.json[key]
        
        elif sub_key in self.json[key]:
            return self.json[key][sub_key]
        return default

    def get_tag(self, key, tag, default=0):
        if TAG_KEY in self.json[key].keys() and tag in self.json[key][TAG_KEY].keys():
            return self.json[key][TAG_KEY][tag]
        return default
    
    def is_valid_region(self, region_key):
        valid = False
        # If tags don't exist empty dictionary is returned
        tag_dict = self.get_value(region_key, TAG_KEY, dict())
        coords = self.get_value(region_key, "coord", [])
        tag = ""
        types = []
        if len(coords) >= MIN_POLY_PTS*2:
            for t in tag_dict.keys():
                if t.startswith(REGION_TAG_START) and self.json[region_key][TAG_KEY][t]:
                    valid = True
                    types.append(t[len(REGION_TAG_START):])
                
     
        if len(types) > 1:
            print('More than one region assigned')
            print(types)
        if len(types) > 0:
            tag = types[0]
        
        type_dict = REGION_TYPE_DICT
        
        if len(tag) > 0:
            if tag in type_dict.keys():
                tag = type_dict[tag]
            else:
                print('not in reguion dict')
                tag = REGION_TYPE_DICT['paragraph']
        return valid, tag
        
    def get_regions(self):
        region_keys = []
        region_types = []
        for k in self.line_keys():
            is_valid, region_type = self.is_valid_region(k)
            if is_valid:
                region_keys.append(k)
                region_types.append(region_type)
            
        return region_keys, region_types

    # Given a coordinates list return xy tuples
    def list_to_xy(self, coord_list):
        xy_list = []
        for ind in range(0, len(coord_list), 2):
            xy_list.append((coord_list[ind], coord_list[ind+1]))
        return xy_list

    def coord_str(self, coord_list):
        str_list = []
        for ind in range(0, len(coord_list), 2):
            x = int(coord_list[ind])
            y = int(coord_list[ind+1])
            # Make corrections in coordinates
            if x < 0:
                #print('x=0')
                x = 0
            if y < 0:
                #print('y=0')
                y = 0
            if x > self.size[0]:
                #print('x=size-1')
                x = self.size[0] - 1
            if y > self.size[1]:
                #print('y=size-1', y, self.size[1])
                y = self.size[1] - 1
            str_list.append(f'{x},{y}')
        return ' '.join(str_list)
    
    
    
    # Given an (x,y) list of tuples, return a flat list
    def xy_to_list(self, tuples_list):
        flat_list = [x for pair in tuples_list for x in pair]
        return flat_list

    
    def belongs_to(self, line, region):
        if 'line_polygon' not in self.json[line].keys():
            line_polygon = self.json[line]["coord"]
            line_polygon = self.list_to_xy(line_polygon)
            self.json[line]['line_polygon'] = line_polygon
        if 'line_polygon' not in self.json[region].keys():
            region_polygon = self.json[region]["coord"]
            region_polygon = self.list_to_xy(region_polygon)
            self.json[region]['line_polygon'] = region_polygon
        
        _, _, percent = poly.percent_intersection(self.size, self.json[region]['line_polygon'] , 
                                                  self.json[line]['line_polygon'])
        if percent > THRESHOLD_PERCENT_AREA:
            return True
            
    def assign_lines_to_regions(self):
        for line_key in self.line_keys():
            for region_key in self.region_keys():
                if self.belongs_to(line_key, region_key):
                    
                     # Only include region within itself if it has text or it is crossed out
                    if (region_key == line_key):
                        #print(region_key, line_key)
                        text = self.get_text(region_key)
                        if len(text) > 0 or self.get_tag(region_key, CROSSED_TAG) == 1:
                            self.line_dict[line_key] = {'assigned': True, 'region': region_key}
                            self.region_dict[region_key]['lines'].append(line_key)
                            #print('appended')
                        
                        else:
                            #print('key', region_key)
                            #print('Tag value', self.get_tag(region_key, CROSSED_TAG))
                             # Don't add the region within itself as it has no text
                            self.line_dict[line_key] = {'assigned': True, 'region': None}
                    else: # for any line other than region itself
                        self.line_dict[line_key] = {'assigned': True, 'region': region_key}
                        self.region_dict[region_key]['lines'].append(line_key)

    # If any line is unassigned, assign it to text region
    def get_text_region(self):
        region = []
        new_key = 'region_text'
        for line_key in self.line_keys():
            if not self.line_dict[line_key]['assigned']:
                self.line_dict[line_key] = {'assigned': True, 'region': new_key}
                region.append(line_key)
        if len(region) > 0:
            self.region_dict[new_key] = {'lines': region, 'type': REGION_TYPE_DICT['paragraph']}
            self.json['region_text'] = dict()
            
    def get_coords_str(self, key, use_region_dict=False):
        if not "coord" in self.json[key].keys():
            return ""
        if not use_region_dict:
            coords = self.json[key]["coord"]
        elif "coord" in self.region_dict[key]:
            coords = self.region_dict[key]["coord"]
        else:
            coords = []
        return self.coord_str(coords)
    
    def get_baseline_str(self, key):
        # This should not be the case
        if not "baseline" in self.json[key].keys():
            return ""
        coords = self.json[key]["baseline"]
        coords = self.xy_to_list(coords)
        return self.coord_str(coords)    

    def get_text(self, key):
        if not "text" in self.json[key]:
            return ""
        return self.json[key]["text"]

    def get_index(self, key):
        if not "index" in self.json[key]:
            return ""
        return str(self.json[key]["index"])
    
    def get_region_order(self):
        
        return

    def get_line_order(self):
        return

    # Get the convex hull of all polygons in line_list
    def get_convex_hull(self, line_list):
        all_pts = []
        all_polys = []
        for line in line_list:
            if 'line_polygon' not in self.json[line].keys():
                line_polygon = self.json[line]["coord"]
                line_polygon = self.list_to_xy(line_polygon)
                self.json[line]['line_polygon'] = line_polygon
            all_pts.extend(self.json[line]['line_polygon'])
            all_polys.append(self.json[line]['line_polygon'])
            
        
        points = np.array(all_pts)
        try:
            hull_pts, alpha, steps = get_concave_hull_shapely(all_polys)
        except Exception as e:
            print('Problem in concave hull', str(e))
            hull = ConvexHull(points)
            hull_pts = np.array(points[hull.vertices])
        
        return list(hull_pts.flatten())

    def sort_all(self):
        self.get_baselines()
        self.sort_regions()
        self.sort_lines_in_all_regions()

    def get_center_poly(self, key):
        if 'line_polygon' not in self.json[key].keys():
            line_polygon = self.json[key]["coord"]
            line_polygon = self.list_to_xy(line_polygon)
            self.json[line]['line_polygon'] = line_polygon
        center = np.mean(self.json[key]['line_polygon'], axis=0)
        return center
 
    def sort_region_list(self, region_list):
    
        if len(region_list) == 0:
            return [], []
        region_centers = [self.get_center_poly(region) for region in region_list]

        sorted_centers = sorted(enumerate(region_centers), key=lambda x: (x[1][1], -x[1][0]))
        sorted_centers_ind = [x[0] for x in sorted_centers]

        for ind, sorted_index in enumerate(sorted_centers_ind):
            region_key = region_list[sorted_index]
            # Don't think we need this
            self.json[region_key]['sorted_group_index'] = ind

        sorted_regions = [region_list[sorted_index] for sorted_index in sorted_centers_ind]
        return sorted_regions, np.mean(region_centers, axis=0)

    def pt_less(self, pt1, pt2):
        if pt1[1] < pt2[1]:
            return True
        if pt1[1] > pt2[1]:
            return False
        # ht is same
        if pt1[0] < pt2[0]:
            return True
        return False



    def sort_regions(self):
        # Divide the page into three groups...main page region, anything above main page, anything below main page
        # Sort within the three groups
        
        region_list = list(self.region_dict.keys())
        main_regions = [r for r in region_list if self.region_dict[r]['type']['name'] in MAIN_PAGE_REGION_NAMES]
        
        sorted_main, main_center = self.sort_region_list(main_regions)
        above_main = [r for r in region_list 
                         if r not in main_regions and 
                            self.pt_less(self.get_center_poly(r), main_center)]

        below_main = [r for r in region_list 
                         if r not in main_regions and 
                            r not in above_main]

        above_main, above_center = self.sort_region_list(above_main)
        below_main, below_center = self.sort_region_list(below_main)
        
        sorted_regions = []
        if len(above_main) > 0:
            sorted_regions.append(above_main)
        if len(sorted_main) > 0:
            sorted_regions.append(sorted_main)
        if len(below_main) > 0:
            sorted_regions.append(below_main)
        return sorted_regions
        
    def sort_lines_in_all_regions(self):
        for region in self.region_dict.keys():
            self.sort_lines_in_region(region)


    def sort_lines_in_region(self, region_key):
        line_starts = [self.json[line]['baseline'][0] for line in self.region_dict[region_key]['lines']]
        sorted_starts = sorted(enumerate(line_starts), key=lambda x: (x[1][1], -x[1][0]))
        sorted_start_ind = [x[0] for x in sorted_starts]

        for ind, sorted_index in enumerate(sorted_start_ind):
            line_key = self.region_dict[region_key]['lines'][sorted_index]
            self.json[line_key]['index'] = ind
    
    def get_baselines(self):
        sort = True
        # get baseline for all lines
        for l, v in self.line_dict.items(): 
            #print('doing line', l)
            if 'line_polygon' not in self.json[l].keys():
                line_polygon = self.json[l]["coord"]
                line_polygon = self.list_to_xy(line_polygon)
                self.json[l]['line_polygon'] = line_polygon
            if self.get_tag(l, ORIENT_TOP_BOTTOM_TAG):
                # REversed = False is top to bottom
                # Reversed = True is bottom to top
                baseline = points.get_vertical_baseline(self.json[l]['line_polygon'], reversed=False)
                sort = False
            elif self.get_tag(l, ORIENT_BOTTOM_TOP_TAG):
                baseline = points.get_vertical_baseline(self.json[l]['line_polygon'], reversed=True)
                sort = False
            else:
                baseline = points.get_baseline_chunks(self.json[l]['line_polygon'])
            # Right to left order
            if self.json[l]['language'] == ARABIC and sort:
                baseline.sort(key=lambda x: x[0], reverse=True)
            self.json[l]['baseline'] = baseline
    # Do the same for regions
        for r, v in self.region_dict.items():
            if 'baseline' in self.json[r].keys():
                continue
            if 'line_polygon' not in self.json[r].keys():
                line_polygon = self.json[r]["coord"]
                line_polygon = self.list_to_xy(line_polygon)
                self.json[r]['line_polygon'] = line_polygon
            baseline = points.get_baseline_chunks(self.json[r]['line_polygon'])
            # Right to left order
            baseline.sort(key=lambda x: x[0], reverse=True)
            self.json[r]['baseline'] = baseline
            
            
    def get_language(self, line_key):
        if 'language' in self.json[line_key].keys():
            return self.json[line_key]['language']
        lang = ARABIC
        tagged_txt = ""
        text = self.get_value(line_key, "text", "")
        if len(text) > 0:
            #lang, tagged_txt = detector.detect_language_and_add_tags(text)
            is_latin = self.detector.is_latin(text)
            if is_latin:
                lang = LATIN
            
        self.json[line_key]['tagged_text'] = tagged_txt
        self.json[line_key]['language'] = lang
        
        return lang

    def get_production(self, line_key):
        bold = False
        italics = False
        if "production" in self.json[line_key].keys():
            return self.json[line_key]["production"]
        production = "handwritten-cursive"
        tag_dict = self.get_value(line_key, TAG_KEY, dict())
        for t in tag_dict.keys():
            if t.startswith(PRINTED_TAG_START) and self.json[line_key][TAG_KEY][t]:
                production = "printed"
                if t == PRINTED_TAG_START + 'Bold':
                    bold = True
                if t == PRINTED_TAG_START + 'Italics':
                    italics = True
            if t == ARCHIVIST_TAG and self.json[line_key][TAG_KEY][t]:
                self.json[line_key][ARCHIVIST_TAG] = True
            
                
                
        self.json[line_key]["production"] = production
        self.json[line_key]["bold"] = bold
        self.json[line_key]["italics"] = italics
        self.json[line_key][NUMERIC] = self.detector.is_numeric(self.get_text(line_key))
        
        return self.json[line_key]["production"]
    
    
    def is_bold(self, line_key):
        return self.json[line_key]["bold"]

    def is_italics(self, line_key):
        return self.json[line_key]["italics"]
    
    
    def is_crossed_out_text(self, line_key):
        
        return self.get_tag(line_key, CROSSED_TAG)
    
    
    def get_reading_direction(self, line_key):
        language = self.get_language(line_key)
        if language == ARABIC:
            return "right-to-left"
            
        return "left-to-right"    
 
    # Not used
    def updated_tagged_text_with_production(self, line_key):
        bold = False
        italics = False
        tag_dict = self.get_value(line_key, TAG_KEY, dict())
        for t in tag_dict.keys():
            if t.startswith(PRINTED_TAG_START) and self.json[line_key][TAG_KEY][t]:
                if t == PRINTED_TAG_START + 'Bold':
                    bold = True
                if t == PRINTED_TAG_START + 'Italics':
                    italics = True
        self.json[line_key]['bold'] = bold
        self.json[line_key]['italics'] = italics
            
    # Not used
    # TODO: Take care of mixed english and arabic
    def get_tagged_text(self, line_key):
        return self.get_text(line_key)
        # Not used
        self.get_language(line_key)
        self.get_production(line_key)
        if self.json[line_key]["production"] == "printed":
            self.updated_tagged_text_with_production(line_key)
        return self.json[line_key]['tagged_text']
                
    def get_id(self, key, element_type):
        return element_type + '_' + self.json[key]['id']

            
class PageXML:
    def __init__(self, page_json, output_file):
        self.json = page_json
        self.root = None
        self.page = None
        self.create_xml_page()
        self.output_file = output_file
    # Dictionary should be (example from Page_XML): 
    # <Metadata><Creator></Creator><Created>2019-10-24T23:14:06</Created>
    #   <LastChange></LastChange></Metadata>
    def create_element_from_dict(self, main_key, dict):
        main_et = ET.Element(main_key)
        for key, value in dict.items():
            element = ET.SubElement(main_et, key)
            element.text = value
        return main_et

    def get_text_equiv(self, key):
        tags = ["Unicode"]
        text = self.json.get_tagged_text(key)        
        text_equiv = ET.Element("TextEquiv")
        parent = text_equiv
        for t in tags:
            element = ET.SubElement(parent, t)
            parent = element
        element.text = text 
        return text_equiv
        
    def get_text_style(self, key):
        (bold, italics, crossed) = (False, False, False)
        
        if self.json.is_bold(key): 
            bold = True
        if self.json.is_italics(key):
            italics = True
        if self.json.is_crossed_out_text(key):
            #print('CROSSED')
            crossed = True
        
       
            
        if not bold and not italics and not crossed:
            return None
        text_style = ET.Element("TextStyle")
        if bold:
            text_style.set('bold', 'true')
        if italics:
            text_style.set('italic', 'true')
        if crossed:
            #text_style.set('underlined', 'true')
            text_style.set('strikethrough', 'true')
            
        return text_style
        
        
    def create_line_element(self, key): #, region_index, line_index):
        #id = f"r_{region_index}_l_{line_index}"
        id = self.json.get_id(key, 'line')
        # TextLine element
        line = ET.Element('TextLine', attrib={"id":id, "primaryLanguage":self.json.get_language(key), 
                                              "production": self.json.get_production(key), 
                                              "readingDirection":self.json.get_reading_direction(key), 
                                              "index": self.json.get_index(key)})
        
        # Add coordinates
        pts = self.json.get_coords_str(key)
        coords = ET.Element('Coords', attrib={"points": pts})        
        line.append(coords)
        
        
        # Add text
        text_equiv = self.get_text_equiv(key)
        line.append(text_equiv)
        text_style = self.get_text_style(key)
        if text_style is not None:
            line.append(text_style)     
        
        # Skip this part if there is no text in line
        if len(self.json.get_text(key)) > 0:
            # ADd baseline: Enter as UserDefined as it is computed algorithmically and is more of a center line for each polygon
            pts_baseline = self.json.get_baseline_str(key)
            center_line = ET.Element('UserDefined')
            center_line_sub = ET.Element("UserAttribute", attrib={"name": "center-line", "value": pts_baseline, 
                                                              "description": "Center line through polygon in reading direction"})
            center_line.append(center_line_sub)
            line.append(center_line)
            
        return line
        
    
    def create_text_region_element(self, key, attribs, prefix_key=''): #, out_index):
        #id = f"r_{out_index}"
        id = self.json.get_id(key, 'region')
        id = prefix_key + id
        region = ET.Element('TextRegion', attrib={"id":id, "type": attribs["subType"]})
        pts = self.json.get_coords_str(key)
        coords = ET.Element('Coords', attrib={"points": pts})
        region.append(coords)
        # Check if this is archivist number
        if self.json.get_value(key, ARCHIVE_NUMBER, default=False):
            region.set('custom', ARCHIVE_NUMBER)
        for ind, l in enumerate(self.json.region_dict[key]['lines']):
            text_line = self.create_line_element(l)#, out_index, ind+1)
            region.append(text_line) 
        return region

    # TODO
    def create_image_region_element(self, key, attribs, out_index):
        return

    # Graphic may contain text
    def create_graphic_region_element(self, key, attribs): #, out_index):
        #id = f"g_{out_index}"
        id = self.json.get_id(key, 'region')
        region = ET.Element('GraphicRegion', attrib={"id":id, "type": attribs["subType"]})
        pts = self.json.get_coords_str(key, use_region_dict=True)
        coords = ET.Element('Coords', attrib={"points": pts})
        region.append(coords)

        # if there are lines in the graphics region
        if len(self.json.region_dict[key]['lines']) > 0:
            dict_to_use = {"mainType": "text", "subType": "paragraph"}
            region_sub = self.create_text_region_element(key, dict_to_use, prefix_key='sub_') #, out_index)
            region.append(region_sub)
        return region    
    
    def create_noise_region_element(self, key): 
        id = self.json.get_id(key, 'region')
        region = ET.Element('NoiseRegion', attrib={"id":id})
        pts = self.json.get_coords_str(key, use_region_dict=False)
        coords = ET.Element('Coords', attrib={"points": pts})
        region.append(coords)
        return region
    
    def create_separator_region_element(self, key):
        id = self.json.get_id(key, 'region')
        region = ET.Element('SeparatorRegion', attrib={"id":id})
        pts = self.json.get_coords_str(key, use_region_dict=False)
        coords = ET.Element('Coords', attrib={"points": pts})
        region.append(coords)
        return region
    
    def create_regions(self):
        for ind, (r, v) in enumerate(self.json.region_dict.items()):
            region = None
            if v['type']['mainType'] == 'text':
                region = self.create_text_region_element(r, v['type'])
            elif v['type']['mainType'] == 'graphic':
                region = self.create_graphic_region_element(r, v['type'])
            elif v['type']['mainType'] == 'noise':
                region = self.create_noise_region_element(r) 
            elif v['type']['mainType'] == 'separator':
                region = self.create_separator_region_element(r) 
                
            if region is not None:
                self.page.append(region)
            else:
                print('Region', r, 'Not added to xml')
            
            
    
    def create_PcGTs(self):
        el = ET.Element("PcGts", 
                        attrib={"xmlns":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
                                  "xmlns:xsi":"http://www.w3.org/2001/XMLSchema-instance",
                                  "xsi:schemaLocation":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"})
        return el
    
    def create_region_reading_order(self):
        reading_order = ET.Element("ReadingOrder")
        OrderedGroup = ET.Element("OrderedGroup", attrib={'id': f'region_order_0'})
        reading_order.append(OrderedGroup)
        # Now add various groups
        for ind, gp in enumerate(self.json.sorted_regions):
            
            OrderedGroupIndexed = ET.Element("OrderedGroupIndexed", attrib={'caption':f"record_{ind}", 'continuation':"false", 
                                                                        'id':f"reading_order_0_{ind}", 'index':f"{ind}"})
            for ind1, key in enumerate(gp):
                RegionRefIndexed = ET.Element("RegionRefIndexed", attrib={'regionRef':self.json.get_id(key, 'region'), 
                                                                      'index': str(ind1)})
                OrderedGroupIndexed.append(RegionRefIndexed)
            OrderedGroup.append(OrderedGroupIndexed)
        
        self.page.append(reading_order)
    
    def create_meta_element(self):
        meta_dict = self.json.meta_dict
        meta_items_dict = self.json.meta_items_dict
        
        meta = self.create_element_from_dict("Metadata", meta_dict)
        for k,v in meta_items_dict.items():
            if k.startswith('other'):
                k = 'other'
            item = ET.Element('MetadataItem', attrib={'type':k, 'value':v})
            meta.append(item)
        
        return meta
    
    #TODO: Add <?xml version="1.0" encoding="UTF-8"?>
    def create_xml_page(self):
        # Create root element
        root = self.create_PcGTs()
        # Create Page and add it after meta
        
        meta = self.create_meta_element()
        root.append(meta)    
        page = ET.SubElement(root, "Page", attrib=self.json.page_dict)
        self.root = root
        self.page = page
        self.create_region_reading_order()
        self.create_regions()
    
    def write_xml(self): 
             
        xml_str = ET.tostring(self.root, 'utf-8')
        pretty_xml = dom.parseString(xml_str)
        
        with open(self.output_file, 'w', encoding="utf-8") as file:
            file.write(pretty_xml.toprettyxml(indent='  '))
        
# Will do a binary search for the best alpha starting with upper_alpha
# Thresh specifies an end condition: upper_alpha-lower_alpha<thresh
# The other end condition os that max_iter is reached
# All polygons should be a list of polygons (not one big list of points)
def get_concave_hull_shapely(all_polygons, max_iter=30, thresh=1e-7):
    # Normalize all points to [0, 1]
    upper_alpha = 2.0
    lower_alpha = 0
    steps = 0
    best_alpha = 1.0
    
    polygon_objs = MultiPolygon([Polygon(poly) for poly in all_polygons])
    while steps < max_iter and upper_alpha-lower_alpha>=thresh:
        all_included = False
        alpha = (lower_alpha + upper_alpha)/2.0
        
        
        hull_pts = shapely.concave_hull(polygon_objs, ratio=alpha)
        
        if type(hull_pts) != shapely.geometry.polygon.Polygon:
            print(type(hull_pts))
            steps += 1
            lower_alpha = alpha
            continue
        
        try:
            all_included = all([hull_pts.contains(poly) for poly in polygon_objs.geoms])
        except Exception as e:
            if alpha == 1.0:
                break
        if all_included:
            best_alpha = alpha
            upper_alpha = alpha
        else:
            lower_alpha = alpha

        steps += 1
    # Compute with best alpha
    hull_pts = shapely.concave_hull(polygon_objs, ratio=best_alpha)
    hull = hull_pts.exterior.coords.xy

    hull_poly = [(x, y) for x,y in zip(hull[0], hull[1])]
    return np.array(hull_poly), best_alpha, steps 



# This is only for concave hull of points NOT concave hull of polygons
# Will do a binary search for the best alpha starting with upper_alpha
# Thresh specifies an end condition: upper_alpha-lower_alpha<thresh
# The other end condition os that max_iter is reached
def get_concave_hull(all_polygons, upper_alpha=16384, max_iter=30, thresh=1e-3):
    # Normalize all points to [0, 1]
    max_pt = np.max(all_polygons, axis=0)
    norm_polygons = [[x[0]/max_pt[0], x[1]/max_pt[1]] for x in all_polygons]
    lower_alpha = 0
    steps = 0
    while steps < max_iter and upper_alpha-lower_alpha>=thresh:
        all_included = False
        alpha = (lower_alpha + upper_alpha)/2.0

        hull_pts = alphashape.alphashape(norm_polygons, alpha)
        if type(hull_pts) != shapely.geometry.polygon.Polygon:
            steps += 1
            upper_alpha = alpha
            continue
        all_included = all([hull_pts.intersects(point) for point in MultiPoint(list(norm_polygons)).geoms])
        if all_included:
            best_alpha = alpha
            lower_alpha = alpha
        else:
            upper_alpha = alpha

        steps += 1
    # Compute with best alpha
    hull_pts = alphashape.alphashape(norm_polygons, best_alpha)    
    hull = hull_pts.exterior.coords.xy
    # Back to normal coordinates
    hull_poly = [(x*max_pt[0], y*max_pt[1]) for x,y in zip(hull[0], hull[1])]
    return np.array(hull_poly), best_alpha, steps    

# Get the name of directory for which xml conversion is required from the first argument
# optionally specify that '_tagged' files are to be converted
if __name__ == "__main__":
    input_dir = None
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        input_dir = os.path.abspath(input_dir)
    if len(sys.argv) > 2:
        TARGET_SUFFIX = sys.argv[2]
    print('target_suffix is ', TARGET_SUFFIX)
    
    if input_dir is None or not os.path.exists(input_dir):
        print("Please provide input directory as first argument")
        sys.exit(1)
        
    files = os.listdir(input_dir)
    files.sort()
    for f in files:
        if not f.lower().endswith('jpg'):
            continue
           
        input_image = f
        json_file = get_json_file(os.path.join(input_dir, f))
        
        if json_file is None:
            print('No json for', f)
            continue
        input_json = os.path.join(input_dir, json_file)
        output_xml = os.path.join(input_dir, input_image[:-3]+'xml')
        #if os.path.exists(output_xml):   
         #   print('Already done', output_xml)
         #   continue
        
        print('Converting ', f)
        json_page = xml.TranscriptionPage(filename=os.path.join(input_dir, input_json), 
                                  imagefile=os.path.join(input_dir, input_image))
        json_page.save()
        xml_obj = xml.PageXML(json_page, output_xml)   
        xml_obj.write_xml()
