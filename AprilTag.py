import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# This uses the ArUco module which is part of most OpenCV-Contrib installs
# ArUco has an AprilTag dictionary built-in!
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# ===== EASY TO MODIFY =====
TAG_IDS = [1, 2]           # Which tag IDs to generate for the tracking sheet
TAG_IDS_BASE = [0]         # Base/reference tag — printed separately at full size
TAG_SIZES_MM = [25, 30]    # Sizes in millimeters for tracking tags
BASE_TAG_SIZE_MM = 150     # Size for tag 0 (15 cm × 15 cm)
PRINT_DPI = 300                   # Print resolution in DPI
# ==========================

# A4 page dimensions: 210 × 297 mm
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
MARGIN_MM = 10  # Margin around page

# Convert A4 to pixels
a4_width_px = int(A4_WIDTH_MM / 25.4 * PRINT_DPI)
a4_height_px = int(A4_HEIGHT_MM / 25.4 * PRINT_DPI)
margin_px = int(MARGIN_MM / 25.4 * PRINT_DPI)

# Create A4 page (white background)
page = Image.new('RGB', (a4_width_px, a4_height_px), color='white')
draw = ImageDraw.Draw(page)

# Try to use a nice font, fall back to default if not available
try:
    font = ImageFont.truetype("arial.ttf", int(20 * PRINT_DPI / 72))
    label_font = ImageFont.truetype("arial.ttf", int(14 * PRINT_DPI / 72))
except:
    font = ImageFont.load_default()
    label_font = font

# Layout: 2 columns (for 2 sizes) × 4 rows (for 4 tag IDs)
x_position = margin_px
y_position = margin_px

# Add title
draw.text((margin_px, margin_px // 2), "AprilTag Markers (IDs 1-4, Sizes 25mm & 30mm)", 
         fill='black', font=label_font)
y_position = margin_px * 3

# Spacing settings (in mm, converted to pixels)
TAG_SPACING_H_MM = 40      # Horizontal spacing between tags
TAG_SPACING_V_MM = 45      # Vertical spacing between rows
LABEL_GAP_MM = 15          # Gap between tag and label
tag_spacing_h_px = int(TAG_SPACING_H_MM / 25.4 * PRINT_DPI)
tag_spacing_v_px = int(TAG_SPACING_V_MM / 25.4 * PRINT_DPI)
label_gap_px = int(LABEL_GAP_MM / 25.4 * PRINT_DPI)

# Generate tags and place on page
for tag_id in TAG_IDS:
    x_col = margin_px
    
    for size_mm in TAG_SIZES_MM:
        # Calculate pixel size
        tag_size_px = int(size_mm / 25.4 * PRINT_DPI)
        
        # Generate tag
        tag_image = np.zeros((tag_size_px, tag_size_px), dtype=np.uint8)
        tag_image = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size_px, tag_image, 1)
        
        # Convert to PIL Image
        tag_pil = Image.fromarray(tag_image, mode='L')
        
        # Paste onto page
        page.paste(tag_pil, (x_col, y_position))
        
        # Add label below tag with more space
        label_text = f"ID{tag_id} - {size_mm}mm"
        draw.text((x_col, y_position + tag_size_px + label_gap_px), label_text, fill='black', font=label_font)
        
        # Move to next column with more spacing
        x_col += tag_size_px + tag_spacing_h_px
        
        # Save individual tags for reference
        filename = f"apriltag_36h11_id{tag_id}_{size_mm}mm.png"
        tag_pil.save(filename)
        print(f"Generated: {filename}")
    
    # Move to next row with more spacing
    tag_size_px = int(TAG_SIZES_MM[0] / 25.4 * PRINT_DPI)
    y_position += tag_size_px + tag_spacing_v_px

# Save A4 sheet for tracking tags (IDs 1 & 2)
page_filename = "AprilTags_A4_Sheet.pdf"
page.save(page_filename, 'PDF', dpi=(PRINT_DPI, PRINT_DPI))
print(f"\nA4 sheet saved as: {page_filename}")
print(f"Page size: {A4_WIDTH_MM}mm × {A4_HEIGHT_MM}mm ({a4_width_px}×{a4_height_px}px @ {PRINT_DPI} DPI)")

# --- Generate tag 0 (base/reference) centred on its own A4 page ---
base_tag_px = int(BASE_TAG_SIZE_MM / 25.4 * PRINT_DPI)

base_page = Image.new('RGB', (a4_width_px, a4_height_px), color='white')
base_draw = ImageDraw.Draw(base_page)

base_tag_img = np.zeros((base_tag_px, base_tag_px), dtype=np.uint8)
base_tag_img = cv2.aruco.generateImageMarker(dictionary, 0, base_tag_px, base_tag_img, 1)
base_tag_pil = Image.fromarray(base_tag_img, mode='L')

# Centre on page
center_x = (a4_width_px - base_tag_px) // 2
center_y = (a4_height_px - base_tag_px) // 2
base_page.paste(base_tag_pil, (center_x, center_y))

# Label
title_text = f"AprilTag ID 0 — {BASE_TAG_SIZE_MM}mm × {BASE_TAG_SIZE_MM}mm  (BASE / REFERENCE)"
base_draw.text((margin_px, margin_px), title_text, fill='black', font=label_font)

base_filename_png = f"apriltag_36h11_id0_{BASE_TAG_SIZE_MM}mm.png"
base_filename_pdf = "AprilTag_ID0_Base_100mm.pdf"
base_tag_pil_full = base_page  # save the full page
base_page.save(base_filename_pdf, 'PDF', dpi=(PRINT_DPI, PRINT_DPI))
base_tag_pil_raw = Image.fromarray(base_tag_img, mode='L')
base_tag_pil_raw.save(base_filename_png)
print(f"Base tag PDF saved as: {base_filename_pdf}")
print(f"Base tag PNG saved as: {base_filename_png}")
print(f"\nAll tags generated successfully!")