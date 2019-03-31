# **P4 Behavioral Cloning** 

---
The P4 project is so-called behavioral cloning, it is just like a simulator game. All we need to do(local machine) is to build the whole environment, run simulator and record the training data again and again because of the trained model crashed the boundaries, tune the learning architechture and apply it on the simulator(again and again, too).

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Build the environment on the local machine
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model(What I use is the jupyter notebook file, so I should submit the ipynb file.)
* drive.py for driving the car in autonomous mode(The file contains on the GitHub, and I never changed it.)
* model.h5 containing a trained convolution neural network (This is the model I've trained.)
* writeup_report.md summarizing the results(That's it.)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

Because I run the code on my local machine, I use jupyter to realize the code. The model.ipynb file contains the code for data importing, visualizing, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model adopt the powerful architecture published by NVIDIA autonomous vehicle team. 
The model contains a normalization layer at the bottom, and then five convolutional layers, and four dense layers.
The code lines are in part 4(in model.ipynb). 

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, I added several dropout layers into the model, which turned out to be effective. See the code in part 4(in model.ipynb), too.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.ipynb, the end of part 4).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with a correction factor(such as 0.2, which is mentioned in the course)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to normalize the dataset, use several convolutional layers with downsampling method, use several fully connected layers.

I have deep feeling about importing and operating the huge amount of images. Since I don't have a GPU, I only can run and tune the model on my local machine.
I've tried several methods about improving the efficiency: dumping the images into pickle file, or using generator. The mothod of saving pickle file, which is just like the file format in project 3 traffic sign classification, maybe efficent running on Colab, but when doing this, memory error or 4 GB limit error occur. Another method is to use generator. It is memory-saving, but has a rather low efficiency. Yet I've tried upload all the recorded data to the google drive(it cost severl hours) so I can use Colab, which provides a free GPU. However, unfortunately, it's not stable and its CPU is not so good that it may take tens or hundreds times to pre-reading the data than on local machine...

Well, my first step was to use a convolution neural network model similar to the LeNet, I thought this model might be appropriate because it got a great score about traffic signs classification.

In order to gauge how well the model was working, I split my image and steering angle data into a training(80%) and validation set(20%). I found that my first model both had a low mean squared error on the training set and on the validation set. This implied that the model was underfitting. The car in the simulator often drove on lane edges.

In order to lower the loss, I flipped the images to augment the data, cropped the images, and I apply all the 3 cameras with correction factors.

After gauging the model, I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add several dropout layers, but it seems helpless. The car in the simulator often drove on lane edges, too.

Then I modified the model(NVIDIA) so that it could perform better. It could be better, but the problem was that the car seemed got a new skill, taking shortcut...

Next, I've tried recollect/record the traing data again and again, especially the data to recovery driving from the sides, wrong ways, and so on. Finally, I realize that maybe the cv2.read is the BGR, but the RGB format should be adopted. After modifying the image pre-process with BGR to RGB, the model ran smoothly in the autonomous mode.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160,320,3 image   							| 
| normalization    	| 160,320,3 image   	|
| Cropping2Dcropping=((70,25),(0,0)))        | 65,320,3 image      |
| Convolution 24,5,5     	|           	|
| RELU					|												|
| Average pooling	2,2    	|  			    	| 
| Dropout      |  prob = 0.5        |
| Convolution 36,5,5     	|           	|
| RELU					|												|
| Average pooling	2,2    	|  			    	| 
| Dropout      |  prob = 0.5             |
| Convolution 48,5,5     	|           	|
| RELU					|												|
| Average pooling	2,2    	|  			    	| 
| Dropout      |  prob = 0.5         |
| Convolution 64,3,3     	|           	|
| RELU					|											  	|
| Dropout      |  prob = 0.5          |
| Convolution 64,3,3     	|             	|
| RELU					|											  	|
| Flatten input     |                     |
| Dense		          | 100         		  	|
| Dropout           |  prob = 0.5        |
| Dense		          | 50         		    	|
| Dropout           |  prob = 0.5        |
| Dense		          | 10        			   |
| Dense		          | 1         			   |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one driving clockwise and counter-clockwise with mouse control.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road.

Then I repeated this process on track one to record three laps and the situation recorvering from running error by comparing the model's performance in the simulator.

To augment the data, I also flipped images and angles.

After the collection process, I had more than 40,000 data points. I then preprocessed the data on my local machine.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by many tests. I used an adam optimizer so that manually training the learning rate wasn't necessary.
