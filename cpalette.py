from collections import defaultdict
from itertools import product
from typing import Any
import sys

import numpy as np
from rich import print as rprint


COLOR_STEPS = list(range(0, 256, 1)) # "1" is do every possible color
enu = enumerate


HexColor = Any


def hsv_to_rgb(hsv):
    """
    Convert HSV color values to RGB.

    Args:
        hsv: Tuple of (H, S, V) values where:
            H is in degrees (0-360)
            S is saturation (0-100%)
            V is value (0-100%)

    Returns:
        tuple: (R, G, B) values as integers (0-255)

    Examples:
        >>> hsv_to_rgb((0, 100, 100))
        (255, 0, 0)  # Pure red
        >>> hsv_to_rgb((120, 100, 100))
        (0, 255, 0)  # Pure green
    """
    h, s, v = hsv

    # Convert to 0-1 scale
    h = h / 360.0
    s = s / 100.0
    v = v / 100.0

    if s == 0:
        # Achromatic (gray)
        r = g = b = v
    else:
        h = h * 6  # sector 0 to 5
        i = int(h)
        f = h - i  # fractional part
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:  # i == 5
            r, g, b = v, p, q

    # Convert to 0-255 scale
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return (r, g, b)


def rgb_to_hsv(rgb):
    """
    Convert RGB values to HSV color space.

    Args:
        rgb: Tuple of (R, G, B) values as integers (0-255)

    Returns:
        tuple: (H, S, V) values where:
            H is in degrees (0-360)
            S is saturation (0-100%)
            V is value (0-100%)

    Examples:
        >>> rgb_to_hsv((255, 0, 0))
        (0, 100, 100)  # Pure red
        >>> rgb_to_hsv((0, 255, 0))
        (120, 100, 100)  # Pure green
    """
    r, g, b = rgb

    # Convert RGB to the range 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Find maximum and minimum values
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin

    # Calculate hue
    if diff == 0:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:  # cmax == b
        h = (60 * ((r - g) / diff) + 240) % 360

    # Calculate saturation
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # Calculate value
    v = cmax * 100

    return (round(h), round(s), round(v))


def hex_to_rgb(hex_color):
    """
    Convert a hexadecimal color string to RGB values.

    Args:
        hex_color: String containing a hex color code (with or without '#')

    Returns:
        tuple: (R, G, B) values as integers (0-255)

    Examples:
        >>> hex_to_rgb("#FF0000")
        (255, 0, 0)
        >>> hex_to_rgb("00FF00")
        (0, 255, 0)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Handle both 3-digit and 6-digit hex formats
    if len(hex_color) == 3:
        # Expand 3-digit hex to 6-digit
        hex_color = ''.join([c*2 for c in hex_color])

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def rgb_to_hex(rgb):
    """
    Convert RGB values to a hexadecimal color code.

    Args:
        rgb: Tuple of (R, G, B) values as integers (0-255)

    Returns:
        str: Hexadecimal color code with '#' prefix

    Examples:
        >>> rgb_to_hex((255, 0, 0))
        '#FF0000'
        >>> rgb_to_hex((0, 255, 0))
        '#00FF00'
    """
    r, g, b = rgb

    # Ensure values are within valid range
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))

    # Convert to hex and format as #RRGGBB
    hex_color = '#{:02X}{:02X}{:02X}'.format(r, g, b)

    return hex_color


def contrast_ratio(rgb1, rgb2):
    """
    Calculate the contrast ratio between two colors according to WCAG 2.0.

    Args:
        rgb1: Tuple of (R, G, B) values for the first color (0-255)
        rgb2: Tuple of (R, G, B) values for the second color (0-255)

    Returns:
        float: The contrast ratio (1:1 to 21:1)
    """
    # Convert RGB to relative luminance
    def get_luminance(rgb):
        # Convert RGB to sRGB
        srgb = [c/255 for c in rgb]

        # Adjust values using the formula
        adjusted = []
        for c in srgb:
            if c <= 0.03928:
                adjusted.append(c / 12.92)
            else:
                adjusted.append(((c + 0.055) / 1.055) ** 2.4)

        # Calculate relative luminance
        return 0.2126 * adjusted[0] + 0.7152 * adjusted[1] + 0.0722 * adjusted[2]

    # Calculate luminance for both colors
    lum1 = get_luminance(rgb1)
    lum2 = get_luminance(rgb2)

    # Calculate contrast ratio
    if lum1 > lum2:
        return (lum1 + 0.05) / (lum2 + 0.05)
    else:
        return (lum2 + 0.05) / (lum1 + 0.05)


def bucket(x, low, high, n_buckets):
    return round(((x - low) / (high - low)) * n_buckets)


def inv_bucket(x, low, high, n_buckets):
    norm_val = x / n_buckets # val between 0.0 and 1.0
    return low + (float(norm_val) * (high - low))


def run(col1: HexColor, min_cont = 2.0, n_columns=35):
    n_cont_bucks = 30
    hue_bg = "#1E1E1E"

    # Collect colors w/ a constrast above threshold
    rgbs = []
    hsvs = []
    conts = []
    for col2 in product(COLOR_STEPS, COLOR_STEPS, COLOR_STEPS):
        contrast = contrast_ratio(col1, col2)
        if contrast < min_cont:
            continue
        rgbs.append(col2)
        hsvs.append(rgb_to_hsv(col2))
        conts.append(contrast)

    # Partition into contrast bands
    contrast_buckets = defaultdict(list)
    cont_low, cont_high = min(conts), max(conts)
    for i, cont in enu(conts):
        buck = bucket(cont, cont_low, cont_high, n_cont_bucks)
        contrast_buckets[buck].append(i)

    # Display Palette
    # - Each row is a contrast level, each column is a hue
    bg_col = rgb_to_hex(col1)
    hue_rows = []
    for key in sorted(contrast_buckets.keys()):
        bucket_idxs = contrast_buckets[key]
        bucket_hsvs = [hsvs[i] for i in bucket_idxs]
        bucket_hsvs.sort(key=lambda x: x[0])

        n_buck_cols = min(n_columns, len(bucket_hsvs))
        idxs = np.linspace(0, len(bucket_hsvs) - 1, n_buck_cols).astype(int)
        row_cont = inv_bucket(key, cont_low, cont_high, n_cont_bucks)
        row = f"{row_cont:.2f}"
        hue_row = row
        for idx in idxs:
            hsv = bucket_hsvs[idx]
            hex_col = rgb_to_hex(hsv_to_rgb(hsv))
            row += f"[{hex_col} on {bg_col}] ■■■[/{hex_col} on {bg_col}]"

            hue_hsv = (hsv[0], 100, 50)
            hue_col = rgb_to_hex(hsv_to_rgb(hue_hsv))
            hue_row += f"[{hue_col} on {hue_bg}] ■■■[/{hue_col} on {hue_bg}]"
        row += f"[{hex_col} on {bg_col}] [/{hex_col} on {bg_col}]{row_cont:.2f}"
        hue_row += f"[{hex_col} on {hue_bg}] [/{hex_col} on {hue_bg}]{row_cont:.2f}"
        hue_rows.append(hue_row)
        rprint(row)

    # Display Hue map
    print()
    for hrow in hue_rows:
        rprint(hrow)


if __name__ == "__main__":
    color_hex = sys.argv[1]
    col1 = hex_to_rgb(color_hex)
    run(col1)
