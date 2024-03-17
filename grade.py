import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
from PIL import ImageFilter
import sys

def threshold_image(img, threshold='mean'):
    if threshold == 'mean':
        thresh_value = np.mean(img)
    else:
        thresh_value = threshold
    binary_img = img > thresh_value
    return binary_img.astype(int)

def get_questions(img, lines):
    new_lines = []
    top = lines[0]
    bottom = lines[2]
    height = bottom - top
    new_lines.append(top)
    prev = top
    direction = "up"
    for i in range(0, 29):
        line = prev + height
        if is_white_row(img, line):
            adjusted_line = adjust_line(line, img, direction=direction)
            if adjusted_line - prev < 10:
                adjusted_line = line
        else:
            adjusted_line = line
        if adjusted_line is None:
            break
        if adjusted_line - prev < 10:
            direction = "down"
            adjusted_line = adjust_line(line, img, direction=direction)
        new_lines.append(adjusted_line)
        prev = adjusted_line
    if len(new_lines) > 1:
        height = new_lines[1] - new_lines[0]
        for i in range(1, 5):
            new_lines.append(prev + i * height)

    return combine_lines(new_lines)


def sobel(img):
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = np.zeros_like(img, dtype=float)
    gradient_y = np.zeros_like(img, dtype=float)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            gradient_x[i, j] = np.sum(sobel_x * img[i-1:i+2, j-1:j+2])
            gradient_y[i, j] = np.sum(sobel_y * img[i-1:i+2, j-1:j+2])


    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    return gradient_magnitude

def combine_lines(lines):
    combined_lines = [lines[0]]
    for i in range(1, len(lines)):
        if lines[i] - lines[i-1] > 10:
            combined_lines.append(lines[i])
    return combined_lines

def is_white_row(img,y):
    # Check if the row is white
    return np.any(img[y, :] > 250)

def adjust_line(y, img, direction='up'):
    max_y = img.shape[0]

    if direction == 'up':
        # Move the line up until a non-white row is found
        while y > 0 and y < max_y - 1 and is_white_row(img,y):
            y -= 1
    elif direction == 'down':
        # Move the line down until a non-white row is found
        while y < max_y - 1 and is_white_row(img,y):
            y += 1
    else:
        raise ValueError("Direction must be 'up' or 'down'.")

    # Additional check to prevent getting stuck on the edge
    if (direction == 'up' and y == 0) or (direction == 'down' and y == max_y - 1):
        return None  # No adjustment possible, already at the edge
    
    return y

def get_question_boxes(img, lines):
    question_boxes = []
    # Assuming there are equal columns and the boxes are centered
    width = img.shape[1]
    col_width = width // 3

    for i in range(len(lines) - 1):
        y_top = lines[i]
        y_bottom = lines[i + 1]
        
        for j in range(3):  # Three columns
            x_left = j * col_width
            x_right = (j + 1) * col_width if j < 2 else width - 20  # Adjust for the last column
            if j == 0:
                x_left += 100
            if j == 1:
                x_right += -130
            if j ==2:
                x_left += -125
                x_right += -200
            question_boxes.append((y_top, y_bottom, x_left, x_right))
    qbs = sort_boxes(question_boxes)
    return qbs

def sort_boxes(boxes):
    boxes = sorted(boxes, key=lambda box: box[0])[:86]
    # draw_boxes_sequence(edge_detected_img, boxes)
    # sort boxes by column
    column1 = []
    column2 = []
    column3 = []
    for i in range(0, len(boxes), 3):
        column1.append(boxes[i])
        column2.append(boxes[i+1])
        column3.append(boxes[i+2])
        if i == 78:
            break
    column1.append(boxes[i+3])
    column2.append(boxes[i+4])
    column1.append(boxes[i+6])
    column2.append(boxes[i+7])
    return column1 + column2 + column3

def draw_boxes_on_image(image, boxes):
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.imshow(image, cmap='gray')
    for box in boxes:
        y_start, y_end, x_start, x_end = box
        # draw lines
        ax.add_patch(Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=None, edgecolor='red'))
    plt.show()

def find_answer(img, boxes):
    section_width = img.shape[1] // 5
    max_avg_intensity = 0
    max_avg_section = 0

    for i in range(5):
        section = img[:, i * section_width : (i + 1) * section_width]
        avg_intensity = np.mean(section)
        if avg_intensity > max_avg_intensity:
            max_avg_intensity = avg_intensity
            max_avg_section = i

    return max_avg_section + 1

def get_vertical_lines(img):
    vertical_sum = np.sum(img, axis=0)
    vertical_lines = []
    threshold = 0.85
    while len(vertical_lines) != 30:
        vertical_threshold = threshold * np.max(vertical_sum)
        vertical_lines = np.where(vertical_sum > vertical_threshold)[0]
        vertical_lines = np.unique(vertical_lines)
        vertical_lines = [vertical_lines[0]] + [vertical_lines[i] for i in range(1, len(vertical_lines)) if vertical_lines[i] - vertical_lines[i-1] > 10]
        vertical_lines = filter_lines(vertical_lines)
        threshold -= 0.01
    return vertical_lines

def get_horizontal_lines(img):
    horizontal_sum = np.sum(img, axis=1)
    horizontal_sum = np.sum(img, axis=1)
    horizontal_threshold = 0.87 * np.max(horizontal_sum)
    horizontal_lines = np.where(horizontal_sum > horizontal_threshold)[0]
    horizontal_lines = np.unique(horizontal_lines)
    horizontal_lines = [horizontal_lines[0]] + [horizontal_lines[i] for i in range(1, len(horizontal_lines)) if horizontal_lines[i] - horizontal_lines[i-1] > 10]
    return horizontal_lines

def filter_lines(lines):
    # remove lines that are too close to each other
    filtered_lines = [lines[0]]
    for i in range(1, len(lines)):
        if lines[i] - lines[i-1] > 15:
            filtered_lines.append(lines[i])
    return filtered_lines

def draw_vertical_lines(img, lines):
    for line in lines:
        cv2.line(img, (line, 0), (line, img.shape[0]), (0, 255, 0), 2)
    cv2.imshow("Vertical Lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_answer_choices(img, boxes, vertical_lines):
    answer_choices = []
    answer_boxes = []
    num = 1
    for box in boxes:
        y_start, y_end, x_start, x_end = box
        curr_answers = []
        black_pixels_list = []
        sections = []
        # Calculate black pixels for each section within the box
        for i in range(0, len(vertical_lines) - 1, 2):
            # Check if these vertical lines are within the box
            if vertical_lines[i] < x_start or vertical_lines[i + 1] > x_end:
                continue
            x1 = vertical_lines[i]
            x2 = vertical_lines[i + 1]
            section = img[y_start:y_end, x1:x2]
            black_pixels = np.sum(section < 100)
            black_pixels_list.append(black_pixels)
            sections.append((y_start, y_end, x1, x2))

        # Calculate average black pixels if there are any sections to analyze
        if black_pixels_list:
            average_black_pixels = np.mean(black_pixels_list)
            # Determine which sections are above average and should be considered an answer choice
            answer_num = 1  # Start numbering answer choices from 1
            for i in range(len(black_pixels_list)):
                if black_pixels_list[i] > average_black_pixels:
                    curr_answers.append(answer_num)
                    answer_boxes.append(sections[i])
                    
                answer_num += 1
        # check if question has written answer next to it
        curr_lines = []
        if num < 30:
            curr_lines = [vertical_lines[0]]
        elif num < 60:
            curr_lines = [vertical_lines[9], vertical_lines[10]]
        else:
            curr_lines = [vertical_lines[19], vertical_lines[20]]
        if check_for_extra(img, box, curr_lines):
            curr_answers.append('x')

        answer_choices.append(curr_answers)
        num += 1
    draw_answers(img, answer_boxes)
    return answer_choices

def draw_answers(img, answers):
    # draw filled in rectangles for each answer
    # save the image to a file using imwrite
    for box in answers:
        y_start, y_end, x_start, x_end = box
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), -1)
    cv2.imwrite("scored.jpg", img)

def draw_boxes_sequence(img, boxes):
    # draw boxes one by one, wait for user input to continue
    for box in boxes:
        y_start, y_end, x_start, x_end = box
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("Boxes", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
        

def draw_lines_and_boxes(img, vertical_lines, question_boxes):
    # draw boxes and lines
    for line in vertical_lines:
        cv2.line(img, (line, 0), (line, img.shape[0]), (0, 255, 0), 2)
    for box in question_boxes:
        y_start, y_end, x_start, x_end = box
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow("Lines and Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_answers(answers):
    answer_map = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 'x': 'x' }
    # if multiple answers are selected, print all of them
    for i, answer in enumerate(answers):
        print(f"{i + 1}: {''.join([answer_map[ans] for ans in answer])}")

def check_for_extra(img, box, lines):
    y_start, y_end, x_start, x_end = box
    if len(lines) == 1:
        section = img[y_start:y_end, 0:lines[0]]
    else:
        section = img[y_start:y_end, lines[0]:lines[1]-5]
    # count number of black pixels, if it is high, return true
    black_pixels = np.sum(section < 100)
    # print(black_pixels)
    return black_pixels > 880


def draw_horizontal_lines(img, lines):
    for line in lines:
        cv2.line(img, (0, line), (img.shape[1], line), (0, 255, 0), 2)
    cv2.imshow("Horizontal Lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_answers_to_file(answers, filename):
    answer_map = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 'x': 'x' }
    # write in format: question number answer (if multiple answers, do not separate)
    with open(filename, 'w') as f:
        for i, answer in enumerate(answers):
            f.write(f"{i + 1} {''.join([answer_map[ans] for ans in answer])}\n")

def check_answers(answers, filename):
    answer_map = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 'x': 'x' }
    # read the correct answers from the file
    # per line, format: question number answer (if multiple answers, do not separate) for example: 1 AB
    with open(filename, 'r') as f:
        correct_answers = f.readlines()
    correct_answers = [line.strip().split() for line in correct_answers]
    correct_answers = { int(answer[0]): answer[1] for answer in correct_answers }
    # compare the answers
    correct = 0
    for i, answer in enumerate(answers):
        curr_answer = ''.join([answer_map[ans] for ans in answer])
        # add question number to front of curr_answer, i+1 is the question number
        if curr_answer == correct_answers[i + 1]:
            correct += 1
        else:
            print(f"Question {i + 1} was incorrect. Answer was {curr_answer}, correct answer was {correct_answers[i + 1]}")
    print(f"Correct answers: {correct}/{len(answers)}")

def process_test(file_name_input, file_name_output):
    print("Recogninzing " + file_name_input + "...")
    img = cv2.imread(file_name_input)
    img = img[660:]
    edge_detected_img = sobel(img)
    binary_img = threshold_image(edge_detected_img, 'mean')
    vertical_lines = get_vertical_lines(binary_img)
    horizontal_lines = get_horizontal_lines(binary_img)
    horizontal_lines = get_questions(edge_detected_img, horizontal_lines)
    question_boxes = get_question_boxes(edge_detected_img, horizontal_lines)
    answers = get_answer_choices(img, question_boxes, vertical_lines)
    print_answers(answers)
    write_answers_to_file(answers, file_name_output)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("error: please give an input image name and output file name as a parameter, like this: \n"
                     "python3 grade.py input.jpg output.txt")
    process_test(sys.argv[1], sys.argv[2])