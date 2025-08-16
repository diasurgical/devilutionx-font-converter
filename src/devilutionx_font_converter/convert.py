import argparse
import dataclasses
import multiprocessing
import os

import freetype
import tqdm
import wand.drawing
import wand.image

_DATADIR = os.path.join(os.path.dirname(__file__), "data")

_STROKE_COLOR = "rgb(19, 11, 0)"


@dataclasses.dataclass(frozen=True)
class DrawSettings:
    stroke_color: str
    stroke_width: int


def get_text_y(m: wand.drawing.FontMetrics, i: int, frame_height: int) -> int:
    return int(m.ascender) + (i * frame_height)


def make_glyphs(
    font_path: str,
    font_size: int,
    draw_settings: DrawSettings,
    tiled_texture: wand.image.Image | None,
    frame_height: int,
    cp_range: tuple[int, int],
    output_directory: str,
):
    with wand.drawing.Drawing() as ctx:
        ctx.font = font_path
        ctx.font_size = font_size

        if tiled_texture is None:
            ctx.stroke_color = draw_settings.stroke_color
            ctx.stroke_width = draw_settings.stroke_width

        code_points = [chr(cp_i) for cp_i in range(cp_range[0], cp_range[1])]
        with wand.image.Image(width=100, height=100) as tmp:
            font_metrics: list[wand.drawing.FontMetrics] = [
                ctx.get_font_metrics(tmp, text=cp) for cp in code_points
            ]
        max_width = int(max(m.text_width for m in font_metrics))

        for i, (cp, m) in enumerate(zip(code_points, font_metrics)):
            ctx.text(x=0, y=get_text_y(m, i, frame_height), body=cp)

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
                    for i, (cp, m) in enumerate(zip(code_points, font_metrics)):
                        stroke_ctx.text(x=0, y=get_text_y(m, i, frame_height), body=cp)
                    stroke_ctx.draw(stroke_img)
                ctx.composite(
                    operator="plus",
                    left=0,
                    top=0,
                    width=max_width,
                    height=frame_height * 256,
                    image=stroke_img,
                )

        group_name = f"{cp_range[0] // 256:04X}"
        output_prefix = os.path.join(output_directory, f"{font_size}-{group_name}")
        with wand.image.Image(width=max_width, height=frame_height * 256) as img:
            ctx.draw(img)
            img.save(filename=f"{output_prefix}.png")

        with open(f"{output_prefix}.txt", "w") as f:
            f.write("\n".join(str(int(m.text_width)) for m in font_metrics) + "\n")


def created_tiled_texture(texture_path: str, frame_height: int) -> wand.image.Image:
    texture = wand.image.Image(filename=texture_path)
    with texture.clone() as tile:
        tile.crop(bottom=frame_height)
        tiled_texture = wand.image.Image(width=100, height=frame_height * 256)
        tiled_texture.texture(tile)
    return tiled_texture


def get_code_point_ranges(
    font_path: str, min_cp: int, max_cp: int
) -> list[tuple[int, int]]:
    ft_font = freetype.Face(font_path)
    ranges = []
    for begin in range(min_cp, max_cp + 1, 256):
        if begin >= 0xD800 and begin <= 0xDFFF:
            # Surrogate range
            continue
        end = min(begin + 256, max_cp + 1)
        if all(ft_font.get_char_index(cp) == 0 for cp in range(begin, end)):
            continue
        ranges.append((begin, end))
    return ranges


_MP_TILED_TEXTURE = None


def mp_init(texture_path: str):
    if not texture_path:
        return
    global _MP_TILED_TEXTURE
    _MP_TILED_TEXTURE = created_tiled_texture(texture_path, 100)


@dataclasses.dataclass(frozen=True)
class MpTask:
    font_path: str
    font_size: int
    frame_height: int
    cp_range: tuple[int, int]
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
    ranges = get_code_point_ranges(font_path, min_cp, max_cp)
    os.makedirs(output_directory, exist_ok=True)

    # Threading doesn't appear to work with wand-py, so we use multiprocessing.
    with multiprocessing.Pool(
        processes=os.cpu_count(),
        initializer=mp_init,
        initargs=[texture_path],
    ) as pool:
        tasks = [
            MpTask(
                font_path,
                font_size,
                frame_height,
                cp_range,
                output_directory,
                draw_settings,
            )
            for cp_range in ranges
        ]
        for _ in tqdm.tqdm(pool.imap_unordered(mp_work, tasks), total=len(ranges)):
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
