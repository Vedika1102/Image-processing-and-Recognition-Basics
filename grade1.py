import sys, os
import numpy as np
from utils import pad_image
from PIL import Image, ImageFilter, ImageDraw

# region functions
def is_box_present(region, border_thickness, filled_threshold):
    top_border = region[:border_thickness, :]
    bottom_border = region[-border_thickness:, :]
    left_border = region[:, :border_thickness]
    right_border = region[:, -border_thickness:]

    # if all borders have enough filled pixels
    if (np.sum(top_border) >= filled_threshold and np.sum(bottom_border) >= filled_threshold and
        np.sum(left_border) >= filled_threshold and np.sum(right_border) >= filled_threshold):
        return True
    else:
        return False

# convert list of filled boxes to answers
def convert_answer_to_text(lst):
    options = ['A','B','C','D','E']
    return ''.join([options[n[2]-1] for n in lst])

# scribbled answers near the box
def check_scribbled(image, box, receptive_field_shape):
    y, x = receptive_field_shape
    px, py = box[1], box[0]
    px -= 90 
    region = image[py-y//2:py+y//2,px-x//2:px+x//2]
    return region.sum() > 50

# taken from another course assignment (E-535)
def dilation(image, structure):
    result = np.zeros_like(image)
    pad_img = pad_image(image) 
    for i in range( image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.max(pad_img[i:i+3, j:j+3] & structure)
    return result

def erosion(image, structure):
    result = np.zeros_like(image)
    pad_img = pad_image(image) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.min(pad_img[i:i+3, j:j+3] | ~structure)
    return result

def opening(image, structure):
    return dilation(erosion(image, structure), structure) # dialation of erosion

def draw_rectangles(draw, boxes, receptive_field_coords, offsets, scribble_box = None):
    ry, rx = receptive_field_coords
    y_off, x_off = offsets
    for box in boxes:
        y, x = box[0], box[1]
        upper_left = (x-rx//2 + x_off, y-ry//2 + y_off)
        bottom_right = (x+rx//2 + x_off, y+ry//2 + y_off)
        draw.rectangle([upper_left, bottom_right], outline=(0,255,0))

    if scribble_box != None:
        y, x = scribble_box[0], scribble_box[1]-90
        upper_left = (x-rx//2 + x_off, y-ry//2 + y_off)
        bottom_right = (x+rx//2 + x_off, y+ry//2 + y_off)
        draw.rectangle([upper_left, bottom_right], outline=(0,255,0))
# endregion functions 

def run(image_path, output_path):
    image = original_image = Image.open(image_path)
    print(f"Successfully loaded {image_path}")
    image = image.convert('L')
    original_image = original_image.convert('RGB')

    # gaussian blur to smooth out the image
    image = image.filter(ImageFilter.GaussianBlur(radius=1)) 
    image = np.array(image) # to numpy
    print("Applied Gaussian blur")

    # thresholding
    image = np.where(image > 150, 255, 0)
    print("Applied Thresholding")

    x_offset = 100
    y_offset = 662
    # crop the relevant part of the image containing MCQs
    image = image[y_offset:,x_offset:-x_offset]//255

    print("Applying Opening, this might take some time")
    # opening
    structure = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
    image = dilation(erosion(image, structure), structure) # dialation of erosion
    
    # invert image so the black edges (0) become 1 which simplifies the calculations
    inverted_img = (1-image)

    ry, rx = (38, 36) # receptive field dimensions
    col_starts = [int(inverted_img.shape[1]/3) * i for i in range(3)]
    col_width = int(inverted_img.shape[1]/3)
    question_count = 0
    results = ['' for _ in range(85)]
    
    draw = ImageDraw.Draw(original_image)

    print("Processing image")
    # each column of the MCQ sheet
    for ind, start in enumerate(col_starts):
        box_count = 0
        filled_boxes = []
        current_x = int(ry / 2) + start
        current_y = int(rx / 2)
        boxes = []

        # each box in the column
        while True:
            region = inverted_img[current_y-ry//2:current_y+ry//2, current_x-rx//2:current_x+rx//2]
            box_present = is_box_present(region, border_thickness=5, filled_threshold=29) 
            #print(current_y-ry//2, current_y+ry//2, current_x-rx//2, current_x+rx//2, box_present)
            if box_present: 
                boxes.append((current_y,current_x))
                box_count += 1

                if region.sum() > 500:
                    filled_boxes.append((current_y, current_x, box_count))
                current_x += 50
            else:
                current_x += 4

            # Failure case 
            if current_x > start + col_width - rx - 1 and box_count < 5:
                current_x = int(rx / 2) + start
                current_y += 2
                box_count = 0
                filled_boxes = []
                boxes = []

            # Success case
            if box_count == 5 or current_x > start + col_width - rx - 1:
                scribbled = "x" if check_scribbled(inverted_img, boxes[0], (ry,rx)) else ""
                results[question_count] = convert_answer_to_text(filled_boxes) + scribbled
                draw_rectangles(draw, filled_boxes, (ry, rx), (y_offset, x_offset), boxes[0] if scribbled == "x" else None)
                question_count += 1
                box_count = 0
                filled_boxes = []
                current_x = boxes[0][1]-2
                current_y += 40
                boxes = []

            # exit condition
            if (ind == 0 and question_count == 29) or (ind == 1 and question_count == 58) or (ind == 2 and question_count) == 85: break
                
    print("Finished processing the image.")

    with open(output_path, 'w') as f:
        for i, ans in enumerate(results, start=1):
            f.write(f"{i} {ans}\n")

    marked_image_path = os.path.splitext(output_path)[0] + '_scored.jpg'
    original_image.save(marked_image_path)
    print(f"Successfully saved marked image at: {marked_image_path}")

    print(f"Sucessfully saved output at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python grade1.py <path/to/source_image.jpg> <path/to/output_file.txt>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        raise FileExistsError(f"{sys.argv[1]} - path does not exist.")

    run(sys.argv[1], sys.argv[2])
