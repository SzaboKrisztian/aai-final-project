import ujson as uj
import os, random, io
import pandas as pd
from PIL import ImageOps, Image, ImageDraw
from itertools import chain

def render_single(data, magnification=4):
    res = 256 * magnification
    image = Image.new('L', (res, res), 0)
    ctx = ImageDraw.Draw(image)
    xs = list(map(lambda stroke: stroke[0], data))
    ys = list(map(lambda stroke: stroke[1], data))
    max_x = max(chain(*xs))
    max_y = max(chain(*ys))
    add_x = lambda n: ((n * magnification) + (((res - 1) - (max_x * magnification)) // 2))
    add_y = lambda n: ((n * magnification) + (((res - 1) - (max_y * magnification)) // 2))
    for stroke in data:
        ctx.line(list(zip(map(add_x, stroke[0]), map(add_y, stroke[1]))),
            fill=255, width=magnification*2)
    image = image.resize((res // magnification, res // magnification),
        resample=Image.ANTIALIAS)
    result = io.BytesIO()
    image.save(result, format='bmp')
    return result.getvalue()

def render_multiple(drawings, magnification=4):
    res = 256 * magnification
    image = Image.new('RGBA', (res, res), (255, 255, 255, 255))
    ctx = ImageDraw.Draw(image)
    for drawing in drawings:
        xs = list(map(lambda stroke: stroke[0], drawing))
        ys = list(map(lambda stroke: stroke[1], drawing))
        max_x = max(chain(*xs))
        max_y = max(chain(*ys))
        add_x = lambda n: ((n * magnification) + (((res - 1) - (max_x * magnification)) // 2))
        add_y = lambda n: ((n * magnification) + (((res - 1) - (max_y * magnification)) // 2))
        for stroke in drawing:
            ctx.line(list(zip(map(add_x, stroke[0]), map(add_y, stroke[1]))),
                fill=(0, 0, 0, 1),
                width=magnification*2)
    image = image.resize((res // magnification, res // magnification),
        resample=Image.ANTIALIAS)
    result = io.BytesIO()
    image.save(result, format='bmp')
    return result.getvalue()

def get_pixel_data(drawing):
    png = render_single(drawing)
    gray = ImageOps.grayscale(Image.open(io.BytesIO(png)))
    return list(map(lambda x: x / 255, list(gray.getdata())))

def get_dataset_files():
    root, dirs, files = list(os.walk('../dataset'))[0]
    return list(map(lambda f: os.path.join(root, f),
        filter(lambda f: f.endswith('.ndjson'), files)))

def extract_random_entries(file, size):
    file_data = [*map(uj.loads, open(file, encoding='utf8'))]
    file_data = [*filter(lambda e: e['recognized'] == True, file_data)]
    all_indexes = list(range(len(file_data)))
    random.shuffle(all_indexes)
    indexes = all_indexes[:size]
    return [file_data[i] for i in indexes]

def generate_pixel_columns(df):
    df['pixels'] = df.apply(lambda row: get_pixel_data(row['drawing']), axis=1)
    split_df = df.pixels.apply(pd.Series).add_prefix('pixel')
    result = pd.concat([df.drop(columns=['pixels']), split_df], axis=1)
    return result
