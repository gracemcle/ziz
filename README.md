Hi. I'll try to explain the code here. 

## Random Forest Classifier for Drone Noise Detection
### acoustic_detection/random_forst_wf
Order of workflow: 
80 20 train test split
Feature extraction of train and test sets separatately
Features: Gammatone and Mel-Frequency Cepstral Coefficients, Zero Crossing Rate and RMS Energy,and Chroma features
k-fold CV, k = 5
Model: randfor.py (Random Forest)
Grid search for hyperparameters

## CNN To Classify MAFAULDA dataset
This was a project I did for LANL under Nathan Debardeleben in 2022. I used a CNN to classify each of the different faults from the dataset. For more details on the proejct, visit my linked in for the LANL presentation <https://www.linkedin.com/in/grace-mclewee/>
I left the code untouched from the folder I dug it out of out for initial reactions and nostalgia. I remember everything being super messy; it was my busiest semseter of my undergraduate degree, and I had this project on top of everything! I'll give a rundown of what I think/can remember about it. 

### lanl_cnn/preprocess/datagen2.py

Every 4th sample is a test sample? I guess I didnt know about sklearn's train_test_split...
Making my own labels for each of the sample!

### lanl_cnn/preprocess/down_sample_fft.py
Downsample, normalize, and FFT...is what I put down, but I don't see normalization happening. Also, these scripts were all over the place.

### lanl_cnn/maincnn.py
Model architecture based off of this paper: [Deep learning for diagnosis and classification of faults in industrial rotating machinery](https://d1wqtxts1xzle7.cloudfront.net/121245527/j.cie.2020.10706020250208-1-vpcizt-libre.pdf?1738998058=&response-content-disposition=inline%3B+filename%3DDeep_learning_for_diagnosis_and_classifi.pdf&Expires=1747111028&Signature=OHBKf93blg7eVfJKFxN0h25QRYVoBkGi~OY-mLHDNa9EgPPA7R-afe-aiF6-Vgu~gWN~ng4EDJHZTzy8bQ7Q9k3SJqvdu4KJkb8mYV7CvUrAXu-3UCZBwKnpWfeXMVH~bJ8JVtYLZfX36UNaaJU6yIYUZ2hgtd~rJd5tJeFuUnjxfT4B7vj8UBEKy~H84lh6uFYpAVmLXMJyInCAxfk-IM1BKs7FQoInkarhyjDuJb2H-GlovtYiXc4b8rG1tyTtXM~rUb~iPE~gr3k~wA1fObdsp0Pkws6Pu-iigYp7ULGyfb6xk8l-GbrKdwwPbW6PYAelwMk9-wfABzIETzX0eA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

