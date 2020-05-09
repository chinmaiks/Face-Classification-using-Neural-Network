# Face-Classification-using-Neural-Network
Face Classification using Neural Network

Welcome to Project 3 of Computer Vision!!!!!

Below is the directory structure:
ckaidab_proj03
	-code
		-model.py
		-constants.py
		-data_extraction.py
	-data
		-FDDB-folds
		-originalPics
	-model_output
	-readme
	-ckaidab_proj03.pdf

For the dataset, goto http://vis-www.cs.umass.edu/fddb/ and click on the link <Original, unannotated set of images> and download it in the data folder. Extract the data. 
Next, click on the link <Face annotations> and download the data. Extract the data. 


Changes that need to made:
1. Goto to the constants file and change the variable DIR_PATH to the directory <ckaidab_proj03> directory.
In my case it was 
DIR_PATH = "Project/3/"

Make sure to end with the "/" like I have done above.

2. Run the data_extraction.py file. The command takes one boolean (True or False) argument which indicates whether the data need to be normalized or not. 

command to run:
python data_extraction.py True

The command adds a new directory in the data directory called output and stores all the training data and testing data required for training the model in the .pkl format. 

3. model.py contains the LeNet architecture code as well as the code to train the model.
The command line consists 4 parameters
a. Learning rate
b. Weight decay
c. Boolean to indicate whether to perform Data Augmentation or not.
d. Boolean to indicate whether to perform data normalization or not. 
command to run:
python model.py 0.001 0.0 True True

If you want to run the above command, make sure the data extraction command is run first for the normalized data or without normalization whichever you want to try.

The above command also adds the directory <model_output> to the project directory which contains the results obtained after running the model.

On running the above mentioned commands corresponding messages are displayed in the logs.
