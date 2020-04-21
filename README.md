# COVID-19
A CNN Model to predict if a person is affected with COVID-19 using the x-ray images. The model has been trained on about 300 images and the accuracy of the model currently is 73.19 %.
<br><br>

# Dataset
https://github.com/ieee8023/covid-chestxray-dataset
<br> <br>

# select_covid_patient_X_ray_images.py
This code finds all x-ray images of patients of COVID-19 and stores selected image to an COVID directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in covid folder
Code can be modified for any combination of selection of images
<br><br>


# select_normal_patient_X_ray_images.py
This code finds all x-ray images of normal patients and stores selected image to an normal-images directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in normal-images folder
Code can be modified for any combination of selection of images
<br><br>


# X-RAY Image (COVID-19)
![INPUTS](/images/covid-19.png)
### [Data Set Image]
<br><br>

# Accuracy
![INPUTS](/images/accuracy_74.PNG)
### [Accuracy: 73.19%]
<br><br>

# Plotting Accuracy
![INPUTS](/images/accuracy.PNG)
### [Accuracy]
<br><br>

# Plotting Loss
![INPUTS](/images/loss.PNG)
### [Loss]

<br><br>
# GetCovidtoCSV.py
GetCovidtoCSV creates a CSV format of data set with resized images of your defined size (Height, Width) <br>
We got labels for every image from metadata.csv and in new creted dataset, first attribute defines the label
label == 1 defines COVID-19 and Label == 0 defines all others <br>
Since Excel contains XFD as last column and XFD equals 16384. Thus we have to look for the resized height and width <br>
In this example we took 40 x 40 with 3 channels making total row size (40x40x3)+1 +1 is added because label is stored as first attribute of image

##### This will be used for some users who want to work with CSV dataset and not the images directly
