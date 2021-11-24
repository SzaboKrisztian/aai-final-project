import ujson as uj
import os, random, io
import pandas as pd
from PIL import ImageOps, Image, ImageDraw, ImageChops
from itertools import chain

def render_single(data, resolution=256, magnification=4, invert_color=False):
    scale_factor = 256 / resolution
    color_bg = 255 if not invert_color else 0
    color_fg = 0 if not invert_color else 255
    res = resolution * magnification
    image = Image.new('L', (res, res), color_bg)
    ctx = ImageDraw.Draw(image)
    xs = list(map(lambda stroke: stroke[0], data))
    ys = list(map(lambda stroke: stroke[1], data))
    max_x = max(chain(*xs))
    max_y = max(chain(*ys))
    add_x = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_x * magnification / scale_factor)) // 2))
    add_y = lambda n: ((n * magnification / scale_factor) + (((res - 1) - (max_y * magnification / scale_factor)) // 2))
    for stroke in data:
        ctx.line(list(zip(map(add_x, stroke[0]), map(add_y, stroke[1]))),
            fill=color_fg, width=magnification)
    image = image.resize((res // magnification, res // magnification),
        resample=Image.ANTIALIAS)
    result = io.BytesIO()
    image.save(result, format='png')
    return result.getvalue()

def render_multiple(drawings, resolution=256, magnification=4, invert_color=False):
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
                fill=color_fg, width=magnification)
        images.append(image)
    target = Image.new('L', (res, res), color_bg)
    for image in images:
        target = ImageChops.multiply(target, image)
    out = target.resize((resolution, resolution),
            resample=Image.ANTIALIAS)
    out = ImageOps.grayscale(out) if not invert_color else ImageChops.invert(ImageOps.grayscale(out))
    result = io.BytesIO()
    out.save(result, format='png')
    return result.getvalue()

def get_pixel_data(drawing, resolution=256, magnification=4, invert_color=False):
    image = render_single(drawing, resolution, magnification, invert_color)
    return list(map(lambda x: x / 255, list(Image.open(io.BytesIO(image)).getdata())))

def get_dataset_files():
    root, dirs, files = list(os.walk('./dataset/'))[0]
    return list(map(lambda f: os.path.join(root, f),
        filter(lambda f: f.endswith('.ndjson'), files)))

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

def generate_pixel_columns(df, resolution=256, magnification=4, invert_color=False):
    df['pixels'] = df.apply(lambda row: get_pixel_data(row['drawing'], resolution, magnification, invert_color), axis=1)
    split_df = df.pixels.apply(pd.Series).add_prefix('pixel')
    return pd.concat([df.drop(columns=['pixels']), split_df], axis=1)
