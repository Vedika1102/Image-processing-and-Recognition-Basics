import numpy as np
import os, sys
from PIL import Image
from utils import get_question_ordering

# region functions
def find_alignment_bars(image, start_row):
    for row in range(start_row, 0, -1):  # scan upwards from start_row
        line = image[row, :image.shape[1]//2] # just consider the first half of the row
        transitions = np.where(np.diff(line == 0))[0]  # transitions from white to black or black to white, this is basically taking first order image derivative and getting locations where it is not 0

        if len(transitions) >= 6:  # 6 transitions

            # bar width - w
            b1_width = transitions[1] - transitions[0]
            b2_width = transitions[3] - transitions[2]
            b3_width = transitions[5] - transitions[4]
            
            # reduce the effect of noise by ensuring the bar widths are relatively consistent
            if np.any(abs(np.diff([b1_width, b2_width, b3_width])) > 1.5): continue
            
            detected_w = int(round(np.mean([b1_width, b2_width, b3_width])))
            
            # inter-question gap - g
            g1 = transitions[2] - transitions[1]
            g2 = transitions[4] - transitions[3]
            detected_g = int(np.mean([g1, g2]))
            
            return detected_w, detected_g, row, transitions[5] + detected_g
        
def decode_row(row_pixels, w, g, scan_start):
    decoded_answers = []
    ind = scan_start
    
    while ind + w*5 <= row_pixels.shape[1]:
        window = row_pixels[:, ind:ind + w*5]/255
        answer = []
        
        for i in range(5):  # for each option
            section = window[:, i*w:(i+1)*w]
            avg_val = np.mean(section)
            #print(segment, avg_val)
            
            if avg_val <= .4: # more black pixels
                answer.append(chr(ord('A') + i)) # decode from bit to alphabet
        
        if not answer:  # no bars detected for a question, mostly reached the end of barcode in current rpw
            break
        
        decoded_answers.append(answer)
        ind += w*5 + g  # start of the next question section
    
    return decoded_answers

def decode_barcode(image, question_ordering):
    shuffled_answers = []
    row = image.shape[0] - 1  # start from the bottom row and go up
    
    while len(shuffled_answers) < 85:
        detected_w, detected_g, row, scan_start = find_alignment_bars(image, row)
        if detected_w is None:
            break  # no more alignment bars found, stop decoding
        
        # decode answers from the current row starting at scan_start
        row_answers = decode_row(image[row-2:row, :], detected_w, detected_g, scan_start)
        if not row_answers:
            break  # if no answers are decoded, assume end of bar code
        
        shuffled_answers.extend(row_answers)
        
        # row position to search for the next set of answers, hardcoded 25 units above
        row -= 25
    
    ordered_answers = [''] * len(question_ordering)
    
    # unshuffle 
    unshuffle_order = np.zeros_like(question_ordering)
    unshuffle_order[question_ordering] = np.arange(len(question_ordering))

    for idx, original_idx in enumerate(unshuffle_order):
        if idx < len(shuffled_answers): ordered_answers[original_idx] = shuffled_answers[idx]
    
    return ordered_answers
# endregion functions

def run(source, output_file):
    print(f"Source Image: {source}")
    print(f"Output File: {output_file}")

    image = Image.open(source).convert('L')
    if image is not None: print(f"Successfully opened {source}, processing further . . .")
    image = np.array(image)
    # threshold the image
    image = np.where(image > 100, 255, 0)

    answers = decode_barcode(image, get_question_ordering())
    with open(output_file, 'w') as f:
        for i, ans in enumerate(answers, start = 1):
            f.write(f"{i} {''.join(ans)}")
            if i != len(answers):
                f.write(f"\n")

    print(f"Output successfully saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract.py <path/to/source_image.jpg> <path/to/output_file.txt>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        raise FileExistsError(f"{sys.argv[1]} - path does not exist.")

    run(sys.argv[1], sys.argv[2])