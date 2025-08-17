import math
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter

# The function below was implemented by Gemini.
# It might not do precisely what it says it does.


def apply_outer_bevel(
    image: Image.Image,
    size: int,
    angle: float,
    altitude: float,
    highlight_color: str,
    shadow_color: str,
    highlight_opacity: float = 1.0,
    shadow_opacity: float = 1.0,
    soften: int = 0,
) -> Image.Image:
    """
    Applies a highly accurate Outer Bevel effect by directly blending highlight
    and shadow colors based on a calculated lighting map.

    This method uses a distance transform to create a superior rounded surface,
    resulting in realistic, non-mitered corners.

    Args:
        image: The input Pillow Image object with a transparency channel (RGBA).
        size: The size/thickness of the bevel in pixels.
        angle: The angle (azimuth) of the light source in degrees. Standard
               convention: 0 is right, 90 is top, 180 is left.
        altitude: The altitude of the light source in degrees (90 is overhead).
        highlight_color: The color of the highlight (e.g., '#FFFFFF').
        shadow_color: The color of the shadow (e.g., '#000000').
        highlight_opacity: Opacity of the highlight from 0.0 to 1.0.
        shadow_opacity: Opacity of the shadow from 0.0 to 1.0.
        soften: The radius for a Gaussian blur to soften the bevel edges.

    Returns:
        A new Pillow Image object with the accurate bevel effect applied.
    """
    # 1. Prepare the base image and its alpha mask
    if image.mode != "RGBA":
        base_image = image.convert("RGBA")
    else:
        base_image = image

    mask = base_image.getchannel("A")
    if soften > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=soften))
    mask_np = np.array(mask, dtype=np.float32)

    # 2. Create a high-quality surface map and the bevel area using a Distance Transform
    dist_inside = distance_transform_edt(mask_np)
    dist_outside = distance_transform_edt(255 - mask_np)

    # The bevel area is defined by pixels outside the shape, up to the 'size' limit.
    bevel_area_np = (dist_outside > 0) & (dist_outside <= size)
    bevel_area_mask = Image.fromarray((bevel_area_np * 255).astype(np.uint8))

    # The surface map is a combination of inside and outside distances.
    surface_map_np = dist_inside - dist_outside
    surface_map_np = np.clip(surface_map_np, -size, size)

    depth_scale = 1.0
    surface_map_np = (surface_map_np / size) * 127.5 * depth_scale

    # 4. Calculate surface normals
    # A slight blur on the float surface map before calculating normals prevents aliasing.
    surface_map_np_blurred = gaussian_filter(surface_map_np, sigma=0.5)

    dzdy = sobel(surface_map_np_blurred, axis=0)
    dzdx = sobel(surface_map_np_blurred, axis=1)

    # 5. Calculate the 3D light vector
    angle_rad = math.radians(angle)
    altitude_rad = math.radians(altitude)
    Lx = np.cos(angle_rad) * np.cos(altitude_rad)
    Ly = -np.sin(angle_rad) * np.cos(altitude_rad)
    Lz = np.sin(altitude_rad)

    # 6. Normalize the surface normal vectors
    norm = np.sqrt(dzdx**2 + dzdy**2 + 1)
    norm[norm == 0] = 1
    Nx = -dzdx / norm
    Ny = -dzdy / norm
    Nz = 1.0 / norm

    # 7. Calculate the lighting map (dot product)
    dot_product = Lx * Nx + Ly * Ny + Lz * Nz
    lighting_map_np = (dot_product + 1) * 0.5

    # 8. Blend highlight and shadow colors and opacities using the lighting map
    h_color_np = np.array(
        [int(highlight_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32
    )
    s_color_np = np.array(
        [int(shadow_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32
    )

    blend_factor = lighting_map_np[:, :, np.newaxis]

    # Blend RGB colors
    bevel_rgb_np = s_color_np * (1 - blend_factor) + h_color_np * blend_factor
    bevel_rgb_img = Image.fromarray(bevel_rgb_np.astype(np.uint8), "RGB")

    # Blend opacities
    bevel_alpha_np = (
        shadow_opacity * (1 - lighting_map_np) + highlight_opacity * lighting_map_np
    ) * 255
    bevel_alpha_img = Image.fromarray(bevel_alpha_np.astype(np.uint8), "L")

    # 9. Create the final bevel layer
    bevel_canvas = bevel_rgb_img.convert("RGBA")
    bevel_canvas.putalpha(bevel_alpha_img)

    # Create a final canvas and paste the blended bevel, masked to the bevel area
    final_bevel_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    final_bevel_layer.paste(bevel_canvas, (0, 0), bevel_area_mask)

    # 10. Combine bevels with the original image
    # Place the bevel layer first, then composite the original image ON TOP.
    final_image = Image.alpha_composite(final_bevel_layer, base_image)
    return final_image
