#!/usr/bin/env python3
"""
Generate PNG icons from SVG for PWA
"""
import os
from PIL import Image, ImageDraw, ImageFont
import io

def create_icon(size, filename):
    """Create a PNG icon with the specified size"""
    # Create image with gradient background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background (approximation)
    for y in range(size):
        # Interpolate between midnight (#013D5A) and herb (#708C69)
        ratio = y / size
        r = int(1 + ratio * (112 - 1))
        g = int(61 + ratio * (140 - 61))
        b = int(90 + ratio * (105 - 90))
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))
    
    # Add rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    corner_radius = size // 8
    mask_draw.rounded_rectangle([0, 0, size, size], corner_radius, fill=255)
    
    # Apply mask
    img.putalpha(mask)
    
    # Add snowflake emoji (simplified as text)
    try:
        # Try to use a system font
        font_size = size // 4
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Add text
    text_y_offset = size // 6
    draw.text((size//2, size//2 - text_y_offset), "❄️", font=font, anchor="mm", fill=(252, 243, 227, 255))
    
    # Add app name
    try:
        small_font_size = size // 12
        small_font = ImageFont.truetype("arial.ttf", small_font_size)
    except:
        small_font = ImageFont.load_default()
    
    draw.text((size//2, size//2 + text_y_offset), "GB", font=small_font, anchor="mm", fill=(189, 211, 206, 255))
    draw.text((size//2, size//2 + text_y_offset + small_font_size), "DETECTOR", font=small_font, anchor="mm", fill=(189, 211, 206, 255))
    
    # Add detection dots
    dot_size = size // 32
    draw.ellipse([size//2 - dot_size, size - size//4 - dot_size, size//2 + dot_size, size - size//4 + dot_size], fill=(244, 162, 88, 200))
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

if __name__ == "__main__":
    # Create icons directory if it doesn't exist
    os.makedirs("icons", exist_ok=True)
    
    # Generate various icon sizes
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    
    for size in sizes:
        create_icon(size, f"icons/icon-{size}x{size}.png")
    
    print("All icons generated successfully!")