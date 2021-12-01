from genericpath import exists
from itsdangerous import exc
import ujson as uj
import os, random, io, joblib, ast
import pandas as pd
from PIL import ImageOps, Image, ImageDraw, ImageChops
from itertools import chain

banned_cats = [
    'squiggle', 'line', 'circle', 'yoga'
]

#rezolutie de 256 pe 256 
#alb-negru e si el de la 0 la 256
#magnifiation is smoothing the lines
#with magnification 4, the drawing will be drawn in the memory 4 times bigger than the resolution, and when it's rendered, it will be drawn 4 times smaller by using the antialias, and that makes the image smoother
def render_single(data, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    scale_factor = 256 / resolution
    color_bg = 255 if not invert_color else 0
    color_fg = 0 if not invert_color else 255
    res = resolution * magnification
    #open a new image in memory
    image = Image.new('L', (res, res), color_bg)
    ctx = ImageDraw.Draw(image)
    #find the max x and max y in order to center the image
    xs = list(map(lambda stroke: stroke[0], data))
    ys = list(map(lambda stroke: stroke[1], data))
    max_x = max(chain(*xs))
    max_y = max(chain(*ys))
    add_x = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_x * magnification / scale_factor)) // 2))
    add_y = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_y * magnification / scale_factor)) // 2))
    #draw
    for stroke in data:
        ctx.line(list(zip(map(add_x, stroke[0]), map(add_y, stroke[1]))),
            fill=color_fg, width=int(magnification * stroke_width_scale))
    #smooth
    image = image.resize((res // magnification, res // magnification),
        resample=Image.ANTIALIAS)
    #create a byte array
    result = io.BytesIO()
    image.save(result, format='png')
    return result.getvalue()

#iterate through all the images and 
def render_multiple(drawings, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    scale_factor = 256 / resolution
    color_bg = 255
    color_fg = 255 - max(255 // len(drawings), 2500//len(drawings))
    res = resolution * magnification
    images = []
    for drawing in drawings:
        image = Image.new('L', (res, res), color_bg)
        ctx = ImageDraw.Draw(image)
        xs = list(map(lambda stroke: stroke[0], drawing))
        ys = list(map(lambda stroke: stroke[1], drawing))
        max_x = max(chain(*xs))
        max_y = max(chain(*ys))
        add_x = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_x * magnification / scale_factor)) // 2))
        add_y = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_y * magnification / scale_factor)) // 2))
        for stroke in drawing:
            ctx.line(list(zip(map(add_x, stroke[0]), map(add_y, stroke[1]))),
                fill=color_fg, width=int(magnification * stroke_width_scale))
        images.append(image)
    target = Image.new('L', (res, res), color_bg)
    for image in images:
        #blend the images together
        target = ImageChops.multiply(target, image)
    out = target.resize((resolution, resolution),
            resample=Image.ANTIALIAS)
    #if you want it inverted, do the pictures white on black
    out = ImageOps.grayscale(out) if not invert_color else ImageChops.invert(ImageOps.grayscale(out))
    result = io.BytesIO()
    out.save(result, format='png')
    return result.getvalue()


def get_pixel_data(drawing, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    image = render_single(drawing, resolution, magnification, invert_color, stroke_width_scale)
    #get an array unidimensional with each pixel having a value between 0 and 1
    return list(map(lambda x: x / 255, list(Image.open(io.BytesIO(image)).getdata())))

def get_dataset_files():
    root, dirs, files = list(os.walk('./dataset/'))[0]
    return list(map(lambda f: os.path.join(root, f),
        filter(lambda f: f.endswith('.ndjson') and f.split('.')[0] not in banned_cats, files)))

#open a file line by line and extract the first n entries
def extract_first_entries(file, size=None, recognized=None):
    with open(file, 'r', encoding='utf8') as f:
        line = f.readline()
        result = []
        while line is not None and len(result) < size if size is not None else True:
            entry = uj.loads(line)
            if recognized is None or entry['recognized'] == recognized:
                result.append(entry)
            line = f.readline()
    return result

#load a whole file and get random entries
#works less performant
def extract_random_entries(file, size=None, recognized=None):
    """ recognized can be None, True, or False """
    file_data = [*map(uj.loads, open(file, encoding='utf8'))]
    if recognized is not None:
        file_data = [*filter(lambda e: e['recognized'] == recognized, file_data)]
    all_indexes = list(range(len(file_data)))
    random.shuffle(all_indexes)
    size = size if size is not None and size <= len(file_data) else len(file_data)
    indexes = all_indexes[:size]
    return [file_data[i] for i in indexes]

def extract_best_entries(file, size=None, recognized=None, descending=True):
    file_data = [*map(uj.loads, open(file, encoding='utf8'))]
    if recognized is not None:
        file_data = [*filter(lambda e: e['recognized'] == recognized, file_data)]
    df = pd.DataFrame.from_dict(file_data, orient='columns')
    df['complexity'] = df.apply(lambda row: complexity_score(row['drawing']), axis=1)
    df = df.sort_values(by=['complexity'], ascending=not descending)
    return df.reset_index(drop=True) if size is not None and size > len(df) else df[:size].reset_index(drop=True)

def generate_pixel_columns(df, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    df['pixels'] = df.apply(lambda row: get_pixel_data(row['drawing'], resolution, magnification, invert_color, stroke_width_scale), axis=1)
    split_df = df.pixels.apply(pd.Series).add_prefix('pixel')
    return pd.concat([df.drop(columns=['pixels']), split_df], axis=1)

#try to load a run by id and return a dictionary which will have a key for data, model, pca and scaler
def load_run(number, verbose=False):
    path = os.path.join(os.getcwd(), 'runs', str(number))
    data = os.path.join(path, 'data')
    img_params = os.path.join(path, 'img_params')
    models = os.path.join(path, 'models')
    pca = os.path.join(path, 'pca')
    scaler = os.path.join(path, 'scaler')
    if not os.path.exists(path):
        raise Exception("Run does not exist")
    if not os.path.exists(img_params):
        print("Warning: the params used to generate pixel data are unknown")
    pca_scaler_present = False
    pca_exists = os.path.exists(pca)
    scaler_exists = os.path.exists(scaler)
    if pca_exists and scaler_exists:
        pca_scaler_present = True
    elif verbose and pca_exists and not scaler_exists:
        print("Warning: found PCA, but scaler missing. Probably corrupt run")
    elif verbose and not pca_exists and scaler_exists:
        print("Warning: found scaler, but PCA missing. Probably corrupt run")
    result = {}
    try:
        result['data'] = pd.read_feather(data)
    except:
        raise Exception("Failed loading data")
    with open(img_params, 'r') as f:
        result['img_params'] = ast.literal_eval(f.readline())
    try:
        result['models'] = joblib.load(models)
    except:
        print("Failed loading models")
    if pca_scaler_present:
        try:
            result['pca'] = joblib.load(pca)
            result['scaler'] = joblib.load(scaler)
        except:
            if verbose:
                print("Warning: either the PCA or scaler failed to load. Very likely this run is corrupt")
    return result

def complexity_score(drawing):
    total_num_points = 0
    for stroke in drawing:
        total_num_points += len(stroke[0])
    return total_num_points
