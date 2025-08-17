# DevilutionX Font Converter

Converts fonts to the DevilutionX font format.

This is a work-in-progress.



Example usage (with [uv](https://github.com/astral-sh/uv)):

```bash
uv run devilutionx-font-converter \
  --font_path ~/Downloads/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf \
  --output_directory=tmp/noto_ja \
  --font_size=12 \
  --frame_height=20
```

Advanced usage:

```bash
uv run devilutionx-font-converter \
  --font_path ~/Downloads/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf \
  --output_directory=tmp/noto_ja_46 \
  --font_size=46 \
  --frame_height=64 \
  --min_cp=0x4e00 \
  --max_cp=0xff00 \
  --stroke_width=3
```
