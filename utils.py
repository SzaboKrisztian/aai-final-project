import ujson as uj
import os, random, io, joblib, ast
import pandas as pd
from PIL import ImageOps, Image, ImageDraw, ImageChops
from itertools import chain
from ast import literal_eval

banned_cats = [
    'squiggle', 'line', 'circle', 'yoga'
]

def render_single(data, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
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
            fill=color_fg, width=int(magnification * stroke_width_scale))
    image = image.resize((res // magnification, res // magnification),
        resample=Image.ANTIALIAS)
    result = io.BytesIO()
    image.save(result, format='png')
    return result.getvalue()

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
        target = ImageChops.multiply(target, image)
    out = target.resize((resolution, resolution),
            resample=Image.ANTIALIAS)
    out = ImageOps.grayscale(out) if not invert_color else ImageChops.invert(ImageOps.grayscale(out))
    result = io.BytesIO()
    out.save(result, format='png')
    return result.getvalue()

def get_pixel_data(drawing, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    image = render_single(drawing, resolution, magnification, invert_color, stroke_width_scale)
    return list(map(lambda x: x / 255, list(Image.open(io.BytesIO(image)).getdata())))

def get_dataset_files():
    root, dirs, files = list(os.walk('./dataset/'))[0]
    return list(map(lambda f: os.path.join(root, f),
        filter(lambda f: f.endswith('.ndjson') and f.split('.')[0] not in banned_cats, files)))

def extract_first_entries(files, size=None, recognized=None):
    result = []
    for file in (files if isinstance(files, list) else [files]):
        with open(file, 'r', encoding='utf8') as f:
            line = f.readline()
            entries = []
            while (len(line) >= 2) and (len(entries) < size if size is not None else True):
                try:
                    entry = uj.loads(line)
                    if recognized is None or entry['recognized'] == recognized:
                        entries.append(entry)
                finally:
                    line = f.readline()
                    if line is None: break
        result.append(entries)
    flat = [item for sublist in result for item in sublist]
    return pd.DataFrame.from_dict(flat)

def extract_random_entries(files, size=None, recognized=None):
    """ recognized can be None, True, or False """
    result = []
    for file in (files if isinstance(files, list) else [files]):
        file_data = [*map(uj.loads, open(file, encoding='utf8'))]
        if recognized is not None:
            file_data = [*filter(lambda e: e['recognized'] == recognized, file_data)]
        all_indexes = list(range(len(file_data)))
        random.shuffle(all_indexes)
        size = size if size is not None and size <= len(file_data) else len(file_data)
        indexes = all_indexes[:size]
        result.append([file_data[i] for i in indexes])
    flat = [item for sublist in result for item in sublist]
    return pd.DataFrame.from_dict(flat)

def extract_best_entries(files, size=None, recognized=None, descending=True, keep_complexity=False, skip_first=0):
    result = []
    for file in (files if isinstance(files, list) else [files]):
        file_data = [*map(uj.loads, open(file, encoding='utf8'))]
        if recognized is not None:
            file_data = [*filter(lambda e: e['recognized'] == recognized, file_data)]
        df = pd.DataFrame.from_dict(file_data, orient='columns')
        df['complexity'] = df.apply(lambda row: complexity_score(row['drawing']), axis=1)
        df = df.sort_values(by=['complexity'], ascending=not descending)
        if not keep_complexity:
            df = df.drop(columns=['complexity'])
        result.append(df[skip_first:] if size is None or size > len(df) else df[skip_first:size + skip_first])
    return pd.concat(result, ignore_index=True).reset_index(drop=True) if len(result) > 1 else result[0].reset_index(drop=True)

def generate_pixel_columns(df, resolution=256, magnification=4, invert_color=False, stroke_width_scale=1):
    df['pixels'] = df.apply(lambda row: get_pixel_data(row['drawing'], resolution, magnification, invert_color, stroke_width_scale), axis=1)
    split_df = df.pixels.apply(pd.Series).add_prefix('pixel')
    return pd.concat([df.drop(columns=['pixels']), split_df], axis=1)

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
    return sum(list(map(lambda stroke: len(stroke[0]), drawing)))

def equalize_by(dataframe: pd.DataFrame, column: str, num_entries = 1000):
    vcs = dataframe[column].value_counts()
    valid_classes = dict(filter(lambda entry: entry[1] >= num_entries, vcs.items()))
    result = []
    for code in valid_classes.keys():
        result.append(dataframe[dataframe['countrycode'] == code].sample(num_entries))
    result = pd.concat(result, ignore_index=True)
    return result.sample(len(result)).reset_index()

def read_stats():
    root, dirs, files = list(os.walk('./countries/'))[0]
    result = {}
    for dir in dirs:
        with open(f"./countries/{dir}/stats", 'r', encoding='utf-8') as file:
            line = file.readline()
            while line is not None:
                try:
                    temp = literal_eval(line)
                    if isinstance(temp, list):
                        break
                except:
                    continue
                finally:
                    line = file.readline()
            result[dir] = temp

    return result

# def evaluate(H, Y, beta=1.0):
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
#     for i in range(len(H)):
#         if H[i]
#     accuracy = (tp + tn) / (tp + fp + fn + tn)
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (fp + tn)
#     precision = tp / (tp + fp)
#     recall = sensitivity
#     f_score = ( (beta**2 + 1) * precision * recall) / (beta**2 * precision + recall)
#     auc = (sensitivity + specificity) / 2
#     youden = sensitivity - (1 - specificity)
#     p_plus = sensitivity / (1 - specificity)
#     p_minus = (1 - sensitivity) / specificity
#     dp = (np.sqrt(3) / np.pi) * (np.log(sensitivity/(1 - sensitivity) + np.log(specificity/(1 - specificity))))