import cairocffi as cairo
import io
from PIL import ImageOps, Image
from itertools import chain

def convert_to_png(data):
    surface = cairo.SVGSurface(None, 256, 256)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, 256, 256)
    ctx.fill()
    ctx.set_line_width(1)
    ctx.set_source_rgb(0, 0, 0)
    xs = list(map(lambda stroke: stroke[0], data))
    ys = list(map(lambda stroke: stroke[1], data))
    max_x = max(chain(*xs))
    max_y = max(chain(*ys))
    add_x = int((255 - max_x) / 2)
    add_y = int((255 - max_y) / 2)
    for stroke in data:
        ctx.move_to(stroke[0][0] + add_x, stroke[1][0] + add_y)
        for i in range(1, len(stroke[0])):
            ctx.line_to(stroke[0][i] + add_x, stroke[1][i] + add_y)
        ctx.stroke()
    surface.flush()
    return surface.write_to_png(None)

def convert_many(drawings):
    surface = cairo.SVGSurface(None, 256, 256)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, 256, 256)
    ctx.fill()
    ctx.set_line_width(1)
    ctx.set_source_rgba(0, 0, 0, max(1/len(drawings), 0.04))
    for drawing in drawings:
        xs = list(map(lambda stroke: stroke[0], drawing))
        ys = list(map(lambda stroke: stroke[1], drawing))
        max_x = max(chain(*xs))
        max_y = max(chain(*ys))
        add_x = int((255 - max_x) / 2)
        add_y = int((255 - max_y) / 2)
        for stroke in drawing:
            ctx.move_to(stroke[0][0] + add_x, stroke[1][0] + add_y)
            for i in range(1, len(stroke[0])):
                ctx.line_to(stroke[0][i] + add_x, stroke[1][i] + add_y)
            ctx.stroke()
    surface.flush()
    return surface.write_to_png(None)

def get_pixel_data(drawing):
    png = convert_to_png(drawing)
    gray = ImageOps.grayscale(Image.open(io.BytesIO(png)))
    return list(map(lambda x: x / 255, list(gray.getdata())))