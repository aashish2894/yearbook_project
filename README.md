# Yearbook
This is code for implementing yearbook project. It has code for geolocation project too but only yearbook project is done.

## Dependencies
 * python
 * numpy
 * sklearn
 * matplotlib + basemap

# Project Folder Structure
```
data
	yearbook
		train
			F
				000001.png
				...
			M
				000001.png
				...
		valid
			...
		test
			...
		yearbook_train.txt
		yearbook_valid.txt
		yearbook_test.txt
model
	TODO: put your final model file in the folder
src
	TODO: modify load and predict function in run.py
	grade.py
	run.py
	util.py
output
	TODO: output the yearbook test file
	yearbook_test_label.txt
```

## Evaluation
### Data setup
Download the data from the link http://www.cs.utexas.edu/~philkr/cs395t/yearbook.zip and store it in data folder as described in the folder structure.

### Models
Train the model and put the model in the `Model` folder

### Running the evaluation
It will give the result based on the baseline 1 which is the median of the training image.
1. For yearbook validation data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=valid
```

2. For Geolocation validation data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=valid
```

### Generating Test Label for project submission
1. For yearbook testing data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=test
```

2. For Geolocation testing data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=test
```

## Submission
1. Put model and generated test_label files in their respective folder.
2. Remove complete data from the data folder.
3. Add readme.md file in your submission.
4. Project should be run from the grade.py file as shown in the evaluation step and should be able to generate the test_label file.
