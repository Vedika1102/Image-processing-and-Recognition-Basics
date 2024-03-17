import numpy as np
import os, sys
from PIL import Image
from utils import read_answers, get_question_ordering

def run(source, answers_file, output_file):
    print(f"Source Image: {source}")
    print(f"Answers File: {answers_file}")
    print(f"Output File: {output_file}")

    # region funtions
    def jumble_answers(answers, ordering):
        jumbled_answers = [''] * len(answers)

        for i, order in enumerate(ordering):
            jumbled_answers[order] = answers[i]
        
        return jumbled_answers
    
    def encode_question(answers):
        # 5-bit binary for 5 options (A-E)
        binary_sequence = ['0'] * 5

        for answer in answers:
            index = ord(answer) - ord('A')  # letter to index (A=0, B=1, C=2, etc.)
            binary_sequence[index] = '1'
        
        return binary_sequence

    def calculate_questions_per_row(image_width, num_questions, w=2, g=4, side_padding=20):
        # w for one question encoding + gap
        question_width = (w*5) + g

        available_width = image_width - 2*side_padding - 3*w - g*w  

        questions_per_row = available_width // question_width
        return min(questions_per_row, num_questions), question_width

    def embed_barcode(image_array, answers, h=10, w=2, g=4, side_padding=20, bottom_padding=10):
        num_questions = len(answers)
        questions_per_row, _ = calculate_questions_per_row(image_array.shape[1], num_questions, w, g, side_padding)

        if questions_per_row == 0:
            raise ValueError("Image too narrow to encode answers.")
        
        rows_needed = np.ceil(num_questions / questions_per_row).astype(int)
        #if rows_needed > 3:
        #    raise ValueError("Too many questions to fit in the allowed number of rows.")
        
        # barcode area
        barcode_height = rows_needed * h + bottom_padding
        barcode_array = np.full((barcode_height, image_array.shape[1]), 255, dtype=np.uint8)  # White background
        
        for row in range(rows_needed):
            start_y = barcode_height - bottom_padding - (row + 1) * h  # Y position for the current row
            
            # add alignment bars 
            for i in range(3):
                x_position = side_padding + i * (w + g)
                barcode_array[start_y:start_y+h, x_position:x_position+w] = 0  # black = 0
            
            x_offset = side_padding + (3 * (w + g))  # Starting X position for encoded answers after the alignment bars and gaps
            
            for q in range(questions_per_row):
                ind = row * questions_per_row + q
                if ind >= num_questions:
                    break
                answer = answers[ind]
                #print(question_ordering[idx], answer)
                encoded_answer = encode_question(answer)
                for _, bit in enumerate(encoded_answer):
                    if bit == '1':
                        barcode_array[start_y:start_y+h, x_offset:x_offset + w] = 0  # Black bar for '1'
                    x_offset += w
                x_offset += g
        
        # Embed the barcode into the original image
        embedding_start = image_array.shape[0] - barcode_height
        image_array[embedding_start:, :] = barcode_array
        
        return image_array
    # endregion functions

    question_ordering = get_question_ordering()

    answers = read_answers(answers_file) 
    answers = jumble_answers(answers, question_ordering)

    image = Image.open(source).convert('L')
    if image is not None: print(f"Successfully opened {source}, processing further . . .")
    image = np.array(image)

    modified_image = embed_barcode(image, answers, h=20, w=5, g=10, side_padding=20, bottom_padding=10)
    image = Image.fromarray(modified_image)

    print("Successfully embedded the answers barcode into the provdided image, Saving . . .")
    image.save(output_file)
    print(f"Sucessfully saved output image at: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inject.py <path/to/source_image.jpg> <path/to/answers.txt> <path/to/output_image.jpg>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        raise FileExistsError(f"{sys.argv[1]} - path does not exist.")
    if not os.path.exists(sys.argv[2]):
        raise FileExistsError(f"{sys.argv[2]} - path does not exist.")

    run(sys.argv[1], sys.argv[2], sys.argv[3])
