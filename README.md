# DevilutionX Font Converter

Converts fonts to the DevilutionX font format.

This is a work-in-progress.

Requires libmagickwand, on Debian / Ubuntu you can run:

```bash
sudo apt-get install libmagickwand-dev
```

Example usage (with [uv](https://github.com/astral-sh/uv)):

```bash
uv run devilutionx-font-converter \
  --font_path ~/Downloads/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf \
  --output_directory=tmp/noto_ja \
  --font_size=12 \
  --frame_height=20
```
