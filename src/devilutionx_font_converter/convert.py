import argparse
import concurrent.futures
import dataclasses
import math
import os

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import freetype
import tqdm

_DATADIR = os.path.join(os.path.dirname(__file__), "data")


@dataclasses.dataclass(frozen=True, slots=True)
class DrawSettings:
    font_size: int
    stroke_color: str
    stroke_width: float
    supersampling: float


@dataclasses.dataclass(frozen=True, slots=True)
class GlyphMetrics:
    text_width: int


def make_glyphs(
    font: PIL.ImageFont.FreeTypeFont,
    cfg: DrawSettings,
    tiled_texture: PIL.Image.Image | None,
    palette: bytes,
    frame_height: int,
    cp_range: tuple[int, int],
    range_metrics: list[GlyphMetrics],
    output_directory: str,
) -> None:
    code_points = [chr(cp_i) for cp_i in range(cp_range[0], cp_range[1])]
    widths = [m.text_width for m in range_metrics]
    img_width = int(max(widths) + 2 * cfg.stroke_width * cfg.supersampling)
    img_height = int(frame_height * cfg.supersampling * 256)
    img = PIL.Image.new("RGBA", (img_width, img_height))
    ctx = PIL.ImageDraw.Draw(img)

    if tiled_texture is None:
        for i, cp in enumerate(code_points):
            if widths[i] == 0:
                continue
            ctx.text(
                (
                    cfg.stroke_width * cfg.supersampling,
                    i * frame_height * cfg.supersampling + cfg.stroke_width,
                ),
                cp,
                font=font,
                stroke_width=cfg.stroke_width * cfg.supersampling,
                stroke_fill=cfg.stroke_color,
            )
    else:
        # Create a mask from the text
        text_mask = PIL.Image.new("RGBA", (img_width, img_height), 0)
        text_mask_ctx = PIL.ImageDraw.Draw(text_mask)
        for i, cp in enumerate(code_points):
            if widths[i] == 0:
                continue
            text_mask_ctx.text(
                (
                    cfg.stroke_width * cfg.supersampling,
                    i * frame_height * cfg.supersampling + cfg.stroke_width,
                ),
                cp,
                font=font,
            )

        stroke_mask = PIL.Image.new("RGBA", (img_width, img_height))
        stroke_mask_ctx = PIL.ImageDraw.Draw(stroke_mask)
        for i, cp in enumerate(code_points):
            if widths[i] == 0:
                continue
            stroke_mask_ctx.text(
                (
                    cfg.stroke_width * cfg.supersampling,
                    i * frame_height * cfg.supersampling + cfg.stroke_width,
                ),
                cp,
                font=font,
                fill=(0, 0, 0, 0),
                stroke_width=cfg.stroke_width * cfg.supersampling,
                stroke_fill=cfg.stroke_color,
            )
        img.paste(cfg.stroke_color, stroke_mask)

    group_name = f"{cp_range[0] // 256:02x}"
    output_prefix = os.path.join(output_directory, f"{cfg.font_size}-{group_name}")

    if cfg.supersampling != 1.0:
        widths = [math.ceil(w / cfg.supersampling) for w in widths]
        img = img.resize(size=(max(widths), frame_height * 256))
        text_mask = text_mask.resize(size=(max(widths), frame_height * 256))

    if tiled_texture is not None:
        img.paste(tiled_texture.crop((0, 0, img.width, img.height)), text_mask)

    img.save(f"{output_prefix}.png")

    convert_rgba_to_pcx_8bit(
        image=img,
        output_path=f"{output_prefix}.pcx",
        palette=palette,
    )

    if not all(w == widths[0] for w in widths):
        with open(f"{output_prefix}.txt", "w") as f:
            f.write(
                "\n".join(str(w + 2 * cfg.stroke_width if w > 0 else 0) for w in widths)
                + "\n"
            )


def convert_rgba_to_pcx_8bit(
    image: PIL.Image.Image,
    output_path: str,
    palette: bytes,
    transparency_threshold: int = 0,
    partial_alpha_background: tuple[int, int, int] = (0, 0, 0),
    palette_transparent_index: int = 1,
) -> None:
    """Converts an RGBA image to 8-bit PCX image with palette.

    Args:
        image: The image to convert.
        output_path: The output file path.
        transparency_threshold: Pixels with alpha <= this threshold will be made fully transparent.
        partial_alpha_background: Partially transparent pixels will be blended with this color.
        palette_transparent_index: The index of the transparent color in `palette`.
    """
    alpha = image.getchannel("A")
    blend_mask = alpha.point(lambda p: 0 if p <= transparency_threshold else p)
    im_rgb = PIL.Image.new("RGB", image.size, partial_alpha_background)
    im_rgb.paste(image, (0, 0), blend_mask)

    # Can't figure out how to quantize images with mask.
    # For now, as a hack, we fill transparent pixels with the RGB color
    # corresponding to the transparent color (bright green) and hope for the best.
    transparent_color_rgb = (
        palette[3 * palette_transparent_index],
        palette[3 * palette_transparent_index + 1],
        palette[3 * palette_transparent_index + 2],
    )
    im_rgb.paste(
        transparent_color_rgb,
        (0, 0),
        alpha.point(lambda p: 0 if p > transparency_threshold else 255),
    )

    # This is a dummy image that only exist for passing the palette to `quantize`.
    im_pal = PIL.Image.new("P", (0, 0), palette_transparent_index)
    im_pal.putpalette(palette)

    quantized = im_rgb.quantize(palette=im_pal)
    quantized.save(output_path, palette=palette, transparency=palette_transparent_index)


def created_tiled_texture(texture_path: str, frame_height: int) -> PIL.Image.Image:
    tile = PIL.Image.open(texture_path)
    height = frame_height * 256
    width = int(height * 1.2)
    tiled_texture = PIL.Image.new("RGBA", (width, height))
    for y in range(0, height, tile.height):
        for x in range(0, width, tile.width):
            if y + tile.height > height or x + tile.width > width:
                tiled_texture.paste(tile.crop((0, 0, width - x, height - y)), (x, y))
            else:
                tiled_texture.paste(tile, (x, y))
    return tiled_texture


def get_code_point_ranges(
    font_path: str, font_size: int, min_cp: int, max_cp: int
) -> tuple[list[tuple[int, int]], list[list[GlyphMetrics]]]:
    ft_font = freetype.Face(font_path)
    ft_font.set_char_size(font_size << 6)

    ranges = []
    ranges_metrics: list[list[GlyphMetrics]] = []
    for begin in range(min_cp, max_cp + 1, 256):
        if begin >= 0xD800 and begin <= 0xDFFF:
            # Surrogate range
            continue
        end = min(begin + 256, max_cp + 1)

        have_glyphs = False
        range_metrics: list[GlyphMetrics] = []
        for cp in range(begin, end):
            idx = ft_font.get_char_index(cp)
            if idx == 0:
                range_metrics.append(GlyphMetrics(text_width=0))
                continue
            have_glyphs = True
            ft_font.load_glyph(idx, freetype.FT_LOAD_RENDER)
            ft_slot: freetype.GlyphSlot = ft_font.glyph
            ft_glyph_metrics: freetype.GlyphMetrics = ft_slot.metrics
            range_metrics.append(
                GlyphMetrics(text_width=math.ceil(ft_glyph_metrics.horiAdvance / 64))
            )

        if have_glyphs:
            ranges.append((begin, end))
            ranges_metrics.append(range_metrics)

    return ranges, ranges_metrics


def convert_font(
    font_path: str,
    output_directory: str,
    texture_path: str,
    frame_height: int,
    min_cp: int,
    max_cp: int,
    cfg: DrawSettings,
) -> None:
    cp_ranges, ranges_metrics = get_code_point_ranges(
        font_path=font_path,
        font_size=round(cfg.font_size * cfg.supersampling),
        min_cp=min_cp,
        max_cp=max_cp,
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(_DATADIR, "palette.pal"), "rb") as f:
        palette = f.read()

    font = PIL.ImageFont.truetype(font_path, cfg.font_size * cfg.supersampling)
    tiled_texture = (
        created_tiled_texture(texture_path, frame_height) if texture_path else None
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                make_glyphs,
                font=font,
                cfg=cfg,
                tiled_texture=tiled_texture,
                palette=palette,
                frame_height=frame_height,
                cp_range=cp_range,
                range_metrics=range_metrics,
                output_directory=output_directory,
            )
            for cp_range, range_metrics in zip(cp_ranges, ranges_metrics)
        ]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(cp_ranges)
        ):
            future.result()


def main_cli() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--font_path", type=str, required=True)
    argparser.add_argument(
        "--texture_path", type=str, default=os.path.join(_DATADIR, "texture.png")
    )
    argparser.add_argument("--output_directory", type=str, required=True)
    argparser.add_argument("--font_size", type=int, required=True)
    argparser.add_argument("--frame_height", type=int, required=True)
    argparser.add_argument("--min_cp", type=lambda x: int(x, 0), default=0)
    argparser.add_argument("--max_cp", type=lambda x: int(x, 0), default=0x110000)
    argparser.add_argument("--stroke_color", type=str, default="rgb(19, 11, 0)")
    argparser.add_argument("--stroke_width", type=float, default=1.0)
    argparser.add_argument("--supersampling", type=float, default=1.0)
    args = argparser.parse_args()

    convert_font(
        font_path=args.font_path,
        texture_path=args.texture_path,
        output_directory=args.output_directory,
        frame_height=args.frame_height,
        min_cp=args.min_cp,
        max_cp=args.max_cp,
        cfg=DrawSettings(
            font_size=args.font_size,
            stroke_color=args.stroke_color,
            stroke_width=args.stroke_width,
            supersampling=args.supersampling,
        ),
    )


if __name__ == "__main__":
    main_cli()
