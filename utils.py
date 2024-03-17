import numpy as np

def read_answers(file_path):
        answers = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    answers.append(list(parts[1]))
        return answers

def get_question_ordering():
    np.random.seed(42) # ensures consistent jumbling order
    question_ordering = np.arange(85)
    np.random.shuffle(question_ordering)
    return question_ordering

def pad_image(image, kernel_size=3):
    return np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
