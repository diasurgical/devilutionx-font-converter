import argparse
import dataclasses
import math
import multiprocessing
import os

import PIL.Image
import freetype
import tqdm
import wand.drawing
import wand.image

_DATADIR = os.path.join(os.path.dirname(__file__), "data")


@dataclasses.dataclass(frozen=True, slots=True)
class DrawSettings:
    stroke_color: str
    stroke_width: int


@dataclasses.dataclass(frozen=True, slots=True)
class FontMetrics:
    ascender: int


@dataclasses.dataclass(frozen=True, slots=True)
class GlyphMetrics:
    text_width: int


def get_text_y(m: FontMetrics, i: int, frame_height: int) -> int:
    return m.ascender + (i * frame_height)


def make_glyphs(
    font_path: str,
    font_size: int,
    draw_settings: DrawSettings,
    tiled_texture: wand.image.Image | None,
    frame_height: int,
    cp_range: tuple[int, int],
    glyph_metrics: list[GlyphMetrics],
    font_metrics: FontMetrics,
    output_directory: str,
):
    with wand.drawing.Drawing() as ctx:
        ctx.font = font_path
        ctx.font_size = font_size

        if tiled_texture is None:
            ctx.stroke_color = draw_settings.stroke_color
            ctx.stroke_width = draw_settings.stroke_width

        code_points = [chr(cp_i) for cp_i in range(cp_range[0], cp_range[1])]
        max_width = (
            max(m.text_width for m in glyph_metrics) + 2 * draw_settings.stroke_width
        )

        for i, cp in enumerate(code_points):
            ctx.text(
                x=draw_settings.stroke_width,
                y=get_text_y(font_metrics, i, frame_height)
                + draw_settings.stroke_width,
                body=cp,
            )

        if tiled_texture is not None:
            ctx.composite(
                operator="src_atop",
                left=0,
                top=0,
                width=max_width,
                height=frame_height * 256,
                image=tiled_texture,
            )

            with wand.image.Image(
                width=max_width, height=frame_height * 256
            ) as stroke_img:
                with wand.drawing.Drawing() as stroke_ctx:
                    stroke_ctx.font = font_path
                    stroke_ctx.font_size = font_size
                    stroke_ctx.fill_color = "none"
                    stroke_ctx.stroke_color = draw_settings.stroke_color
                    stroke_ctx.stroke_width = draw_settings.stroke_width
                    for i, cp in enumerate(code_points):
                        stroke_ctx.text(
                            x=draw_settings.stroke_width,
                            y=get_text_y(font_metrics, i, frame_height)
                            + draw_settings.stroke_width,
                            body=cp,
                        )
                    stroke_ctx.draw(stroke_img)
                ctx.composite(
                    operator="plus",
                    left=0,
                    top=0,
                    width=0,  # use existing width
                    height=0,  # use existing height
                    image=stroke_img,
                )

        group_name = f"{cp_range[0] // 256:04X}"
        output_prefix = os.path.join(output_directory, f"{font_size}-{group_name}")
        with wand.image.Image(width=max_width, height=frame_height * 256) as img:
            ctx.draw(img)
            img.save(filename=f"{output_prefix}.png")

        convert_png_rgba_to_pcx_8bit(output_prefix)

        with open(f"{output_prefix}.txt", "w") as f:
            f.write(
                "\n".join(
                    str(m.text_width + 2 * draw_settings.stroke_width)
                    for m in glyph_metrics
                )
                + "\n"
            )


def convert_png_rgba_to_pcx_8bit(
    path_prefix: str,
    transparency_threshold: int = 0,
    partial_alpha_background: tuple[int, int, int] = (0, 0, 0),
    palette_transparent_index: int = 1,
):
    """Converts a PNG image to 8-bit PCX image with _PALETTE.

    Args:
        path_prefix: basename of the PNG file (without extension).
        transparency_threshold: Pixels with alpha <= this threshold will be made fully transparent.
        partial_alpha_background: Partially transparent pixels will be blended with this color.
        palette_transparent_index: The index of the transparent color in `_PALETTE`.
    """
    palette = _PALETTE

    im = PIL.Image.open(f"{path_prefix}.png")

    if im.getbands().count != 4:
        im = im.convert("RGBA")

    alpha = im.getchannel("A")
    blend_mask = alpha.point(lambda p: 0 if p <= transparency_threshold else p)
    im_rgb = PIL.Image.new("RGB", im.size, partial_alpha_background)
    im_rgb.paste(im, (0, 0), blend_mask)

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
    quantized.save(
        f"{path_prefix}.pcx", palette=palette, transparency=palette_transparent_index
    )


def created_tiled_texture(texture_path: str, frame_height: int) -> wand.image.Image:
    texture = wand.image.Image(filename=texture_path)
    with texture.clone() as tile:
        tile.crop(bottom=frame_height)
        tiled_texture = wand.image.Image(width=100, height=frame_height * 256)
        tiled_texture.texture(tile)
    return tiled_texture


def get_code_point_ranges(
    font_path: str, font_size: int, min_cp: int, max_cp: int
) -> tuple[FontMetrics, list[tuple[int, int]], list[list[GlyphMetrics]]]:
    ft_font = freetype.Face(font_path)
    ft_font.set_char_size(font_size << 6)

    ranges = []
    ranges_metrics: list[list[FontMetrics]] = []
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

    ft_size_metrics: freetype.SizeMetrics = ft_font.size
    font_metrics = FontMetrics(
        ascender=math.ceil(
            freetype.FT_MulFix(ft_size_metrics.ascender, ft_size_metrics.y_scale) / 64
        ),
    )

    return font_metrics, ranges, ranges_metrics


_PALETTE = None
_MP_TILED_TEXTURE = None


def mp_init(texture_path: str):
    if not texture_path:
        return
    global _MP_TILED_TEXTURE
    _MP_TILED_TEXTURE = created_tiled_texture(texture_path, 100)


@dataclasses.dataclass(frozen=True, slots=True)
class MpTask:
    font_path: str
    font_size: int
    frame_height: int
    cp_range: tuple[int, int]
    glyph_metrics: list[GlyphMetrics]
    font_metrics: FontMetrics
    output_directory: str
    draw_settings: DrawSettings


def mp_work(task: MpTask):
    make_glyphs(
        font_path=task.font_path,
        font_size=task.font_size,
        draw_settings=task.draw_settings,
        tiled_texture=_MP_TILED_TEXTURE,
        frame_height=task.frame_height,
        cp_range=task.cp_range,
        glyph_metrics=task.glyph_metrics,
        font_metrics=task.font_metrics,
        output_directory=task.output_directory,
    )


def convert_font(
    font_path: str,
    output_directory: str,
    texture_path: str,
    font_size: int,
    frame_height: int,
    min_cp: int,
    max_cp: int,
    draw_settings: DrawSettings,
) -> None:
    font_metrics, cp_ranges, glyph_metrics = get_code_point_ranges(
        font_path=font_path, font_size=font_size, min_cp=min_cp, max_cp=max_cp
    )
    os.makedirs(output_directory, exist_ok=True)

    global _PALETTE
    with open(os.path.join(_DATADIR, "palette.pal"), "rb") as f:
        _PALETTE = f.read()

    # Threading doesn't appear to work with wand-py, so we use multiprocessing.
    with multiprocessing.Pool(
        processes=os.cpu_count(),
        initializer=mp_init,
        initargs=[texture_path],
    ) as pool:
        tasks = [
            MpTask(
                font_path=font_path,
                font_size=font_size,
                frame_height=frame_height,
                cp_range=cp_range,
                glyph_metrics=range_glyph_metrics,
                font_metrics=font_metrics,
                output_directory=output_directory,
                draw_settings=draw_settings,
            )
            for cp_range, range_glyph_metrics in zip(cp_ranges, glyph_metrics)
        ]
        for _ in tqdm.tqdm(pool.imap_unordered(mp_work, tasks), total=len(cp_ranges)):
            pass


def main_cli() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--font_path", type=str, required=True)
    argparser.add_argument(
        "--texture_path", type=str, default=os.path.join(_DATADIR, "texture.png")
    )
    argparser.add_argument("--output_directory", type=str, required=True)
    argparser.add_argument("--font_size", type=int, required=True)
    argparser.add_argument("--frame_height", type=int, required=True)
    argparser.add_argument("--min_cp", type=int, default=0)
    argparser.add_argument("--max_cp", type=int, default=0x110000)
    argparser.add_argument("--stroke_color", type=str, default="rgb(19, 11, 0)")
    argparser.add_argument("--stroke_width", type=int, default=1)
    args = argparser.parse_args()

    convert_font(
        font_path=args.font_path,
        texture_path=args.texture_path,
        output_directory=args.output_directory,
        font_size=args.font_size,
        frame_height=args.frame_height,
        min_cp=args.min_cp,
        max_cp=args.max_cp,
        draw_settings=DrawSettings(
            stroke_color=args.stroke_color,
            stroke_width=args.stroke_width,
        ),
    )


if __name__ == "__main__":
    main_cli()
