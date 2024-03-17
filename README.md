# Image processing and Recognition Basics

# Overview:

Our task is to develop a custom automatic grading program for a unique exam scenario. By leveraging computer vision techniques, we aim to accurately and robustly identify student-marked answers on scanned answer sheets. Furthermore, we will explore innovative methods to encode and decode correct answers on these sheets, ensuring a cheating-proof exam environment.

# How to run the code:

For running grade.py:

The scripts are intended to rum from command line, where the user should provide path to the input image name and output file name as a parameter.

python3 ./grade.py input.jpg output.txt

For running inject.py and extract.py:

The scripts are intended to rum from command line, where the user should provide path to the source image, answers file and output file as arguments.

python inject.py source_image.jpg answers.txt output_image.jpg

This command will read the answers from the provided file and jumbles them using a predetermined ordering. The source image is loaded, and the answers are embedded into it using the barcode encoding technique. The modified image is saved to the output file.

python extract.py source_image.jpg output_answers.txt

This command will load the source image, preprocess it, detect the barcode and decodes the barcode to extract the answers. Further the extracted answers are written to the output file.


# Assumptions: 

While addressing the problem statement, we have made a few assumptions to ensure the system operates effectively:

*   Quality and Format of Scanned Sheets: We assume that scanned answer sheets are of high quality and uniform format, with clear markings and minimal distortions. This is crucial for accurate detection of marked answers and barcode data.

*   Marking Scheme Uniformity: The system presupposes a standardized marking scheme across answer sheets, facilitating the detection of answers and interpretation of barcode-encoded keys.

*  Barcode Integrity: For the barcode method to work seamlessly, it's assumed that the barcodes printed on the sheets maintain their integrity post-scanning, allowing for reliable decoding of the correct answers. After scanning the barcode injected sheet, the width ratio of the barcode elements should remain consistent across the bar code. 


# Implementation:


## For grade.py:

### Approach 1 (grade.py): Our approach for extracting marked answers from scanned answer sheets involves several key steps:

1. Image Preprocessing: Utilizes Sobel filters for edge detection, transforming the image to highlight structural details, especially the boundaries of answer boxes.

2. Thresholding: Applies a threshold based on the mean pixel value to convert the image into a binary format, facilitating easier identification of marked areas.

3. Line Detection: Identifies horizontal and vertical lines by analyzing pixel intensity sums across the image. This step is crucial for locating answer boxes by distinguishing between the high contrast areas of marks versus the background.

4. Box Identification: Dynamically calculates the positions of answer boxes by examining the spacing between detected lines, adjusting for variations in the form layout.

5. Answer Extraction: Segments the image into individual boxes and assesses the pixel intensity within each to determine which options have been marked by the student.

6. Answer Mapping: Translates the extracted information into a structured text format, assigning each detected mark to the corresponding question and option.

7. Accuracy Check: Optionally compares extracted answers against a "ground truth" file to evaluate the system's accuracy, providing insights into the effectiveness of the extraction process.


### Approach 2 (grade1.py): This approach considers the answer sheets as binary images and disrectly assumes the black pixels to be edges. And simply uses a box-shape filter(s) to identify the boxes and filled answers.

1. The image is smoothed/denoised using a Gasussian filter, thresholded and "opened". It is further binarized and inverted so the black pixels become 1 which simplifies the calculation. We also trim the image to consider on the area which contains the boxes.

2. We use a box filter of a shape similar to that of the boxes, or rather we "see" the 4 borders of the boxes from a center pixel. Pixel values at each border are summed up and if they exceed a certain threshold, we assume there is a border and if all 4 borders are present, we consider there is a box at the said location. (this is similar to Correlation operation)

3. If the sum of all pixels within the box also exceeds another threshold, we consider the box to be filled, and note the box number.

4. Once all 5 boxes are found for a question, we look 90 pixels to the left of the first box and in a similar manner check if all pixels are above a threshold. If yes, we consider it to be a handwritten answer.

Note that this approach is not scale-invariant and will need to be re-tuned if the scale of the images changes.

## For inject.py:

1. Jumbling Answers: We begin by the jumble_answers function which rearranges the answers based on the randomized order, to prevent the answers from appearing in a predictable sequence. utils.py stores the np.random seed which effectively controls this ordering.

2. Encoding answers: The encode_question function converts each answer choice into a 5-bit one-hot encoding corresponding to options A-E. Each answer choice (A, B, C, D, E) is represented by a binary digit, where '1' indicates the presence of the choice and '0' indicates its absence.

3. Calculating Barcode Size: The calculate_questions_per_row function determines how many questions can be encoded in one row of a barcode, based on the image width. The barcode is split into another row on top of the current row if all the questions cannot fit in one row.

4. Embedding Barcode: The embed_barcode function adds a barcode to the bottom of an image (which is an answer sheet), encoding the answers into the barcode using black and white bars to represent binary data. Each bar is of a specific bar-width 'w' and after each w x 5 pixels there is a gap of 'g' pixels (inter question-gap) which helps seperate the encoding of one question form the other.

5. Alignment Bars: Three alignment bars are added at the beginning of the barcode row. These help to determin the start of the barcode and calculate the bar-width and interquestion-gap and the print-scan process.

## For extract.py:

1. Finding the alignment bars: Three alignment bars are used to infer the bar-width and inter-question gap in a scanned image. The image is scanned bottom to top until these bars are located.

2. Decoding the Barcode: a row of pixels of a certain height, next the the alignment bars is scannned using a sliding window approach. The precalculated bar-width (w) and inter-question gap (g) values are used to parse the barcode, and the one-hot encoded answers are converted back to their alphabet forms. It is due to this sliding window approach we require the barcode width and gap ratio to remain consistent throughout the length barcode.

3. Unshuffling the Answers: The hardcoded jumbling order is used to unshuffle the decoded answers into the correct order and write it to the output file.

# Accuracy on test images:

From the given set of 8 test images, from which we were provided a groundtruth for 2 of the images. The test_images file contain the revised scanned output for every test image and its corresponding groundtruth. 

Quantitative Analysis:

It can be difficult for the model to correctly classify questions with or without marks beside them (in case of coloring multiple boxes). The model can successfully detect each question with writing beside it, however it also incorrectly classifies some questions as having marks next to them when they do not. This is a better problem to have than the opposite as we do not have any false negatives. Also, the model has trouble when some boxes are lightly colored in and other boxes are colored in completely.

* There are in total 8 set of test images, each containing 85 question, from our analysis, there were 26 wrongly written in the text document.
* The correctly identified questions are: 680-26 = 654
* Accuracy: 96.18 %

Qualitative Analysis:

* Error rate: 3.82 %

# Approach:

For solving the inject and extract part of the problem statement, we tried to use the approach of watermarking which would make markings somewhere around the questions or which could make some pixel patterns to encodes the answers, either in the pixel values or in the gradients between them, but all of them can fail due to print-scan related invariances which can distort pixel intensities or gradients. The method of hiding answers in frequency domain was also explored but and was deemed infeasible due to the same reasons. Therefore, for the encodings to survive the print-scan variances we chose the barcode technique, assuming the structure will more ore less remain intact. To do so, we have presented each question in a one hot encoding of 5 bits, each bit representing the options from A-E. If a bit is set to 1, that particular option will be the answer for that particular question. For example: for question with answer b, you'll get 01000. Further we converted these binary encodings into bars, each bar 5px wide therefore each question taking upto 25px, and there is 10px gap between two questions.

### Concerns:

1. If we are able to read the barcode then the test candidate can also crack it.
2. How to figure out where the barcode starts?
3. What if the bar widths are distorted after printing and scanning?

To solve the above concerns we took the following approach:

* To solve the first concern we shuffle the questions, so even if the candidate knows how to read the bar code, they don't know which section corresponds to which question. The key to the correct ordering is hard coded in the program.
* For next two issues, we have added 3 "alignment bars" before the bar code. They are of the same width as the other bars and the distance between them gives the inter question gap. So if the program can use these to infer those values, and next is just to use a sliding window to decode the answers and unshuffle them.


Improvements for future:


* The current implementation assumes a fixed barcode width, which may not be suitable for all images or scanning conditions. We can work on implementing of an algorithm to detect the barcode width adaptively based on the characteristics of the image.
* As the current implementation we scan the entire image to locate alignment bars, we can introduce a preprocessing step to identify and mark the region of interest(ROI) containing the barcode.
