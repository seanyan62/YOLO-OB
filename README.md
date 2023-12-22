# YOLO-OB
This repo is the implementation of
['YOLO-OB: An improved anchor-free real-time multi-scale colon polyp detector in colonoscopy']

The deep learning-based object detection algorithm is currently one of the most effective methods for improving polyp detection. A new model called YOLO-OB is proposed in this repository. Specifically, a bidirectional multiscale feature fusion structure, BiSPFPN, has been developed to enhance the feature fusion capability across different depths of a convolutional neural network. The ObjectBox detection head has been employed, which utilizes a center-based anchor-free box regression strategy to detect polyps of different sizes on feature maps of any scale.

### **1. Data Preparation**
The study used the SUN dataset, which is publicly available, and our self-collected dataset, named Union. Due to potential conflicts of interest, the Union dataset cannot be made publicly accessible. Nevertheless, interested readers can request access to the SUN dataset to conduct further experiments.



#### 1.1. Request SUN dataset and Download
> **Note:** The images used to train YOLO-OB are selected from [SUN dataset](http://amed8k.sundatabase.org), while we could not distribute the original data due to the strict license. 

So first, you need to request the origin colonoscopy video frame from them. In this step, you should download the polyp samples of 100 cases from the links provided by the SUN dataset. 

- **Request for video frames from SUN:** Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself. 
- **Please use your educational email to apply for it and claim it without any commericial purpose.** Thank you for your understanding!


#### 1.2: Unzip SUN dataset and reorganize directory structure.
As for video frames in SUN dataset, these are two groups of samples (positive and negative), which are divided into multiple compressed file format as zip files. We only used images containing polyp samples (positive part), and did not use video frames without polyps (negative part). So we only need to decompress the sundatabase_positive_part. To decompress the zip file downloaded, please input the password provided by origin authors of SUN dataset (i.e., the same as the password that you used for login). 

- **Prepare the positive cases images in SUN**
    - create directory: `mkdir ./data/SUN/`
    - unzip positive cases: `unzip -P sun_password -d ./SUN sundatabase_positive_part\*`, which will take up 11.5 + 9.7 GB of storage space. Please ensure your server has enough space to storage them, otherwise it will fail. Please replace the `sun_password` what you get from SUN's authors.
    - check if correct: `find ./SUN -type f -name "*.jpg" | wc -l`, which should output 49,136 in your terminal.

- **Prepare the new annotations:**
	- We have reorganized the object detection labels of SUN dataset. Next, you need to use "new_labels.zip" from this repository.
	- unzip annotations: `unzip new_labels.zip -d ./SUN`
	- check if correct: `find ./new_labels -type f -name "*.txt" | wc -l`, which should output 49,136 in your terminal.

- **Prepare the code**
	- Download this repository and rename it as "YOLO-OB", then place it in the same directory level as the 'data' folder.
	- Due to disk limitations on my server, I have separated the dataset and code storage. Therefore, the image paths in '`YOLO_OB/config/sun_train.txt`' and '`YOLO_OB/config/sun_valid.txt`' use the absolute path of my local server. Readers are requested to replace them with their corresponding dataset storage locations.


After prepare all the files, your file structure will be similar as below:

```
├──data
    ├──SUN
        ├──images
            ├──case1
                ├──image_name.jpg
                |...
            ├──case2
            |...
            ├──case100
        ├──new_labels
            ├──case1
                ├──image_name.txt
                |...
            ├──case2
            |...
            ├──case100
├──YOLO-OB
    ├──config
    ├──lib
    ├──utils
    ├──detect.py
    ├──models.py
    ├──README.md
    ├──requirements.txt
    ├──test.py
    ├──train.py
```


### **2. Code usage instructions.**

#### **2.1 Install dependencies.**
It is best to use a new virtual environment. Activate the virtual environment, then use the following code to install the software packages required by this project.
```
pip install -r requirements.txt
```


#### **2.2 Training.**
The first step is to confirm the settings in ```Config.cfg```, all the configurations including model_structure, learning rate, batch size and etc. are in it. Secondly, you need to check some additional options when training the model, such as the data directory, the number of epochs, pretrained_weights. These parameters are located in the '`run`' function at the end of '`train.py`'. Once you have finished the above, simply run the code below to start training the model.
```
python train.py
```

#### **2.3 Testing.** 
When you finish training, you can use '`test.py`' to calculate the detection performance of the model. You need to modify the '-w' parameter in the 'run' function at the end of '`test.py`' and specify it as the trained model file.
```
python test.py -w 'path_to_model_file_***.pth'
```

#### **2.4 Visualization of polyp detection.** 
You can use '`detect.py`' to visualize the detection results, which allows you to mark the location of polyps on the image with rectangular boxes. You need to specify three parameters in the '`detect.py`' file's '`run`' function: '`-w`' specifies the trained model file; '`-i`' specifies the path of the original input colonoscopy image; '`-o`' specifies the path to save the generated image.
```
python detect.py -m 'path_to_model_file_***.pth' -i 'path_to_input_images_directory' -o 'path_to_output_images_directory' 
```

#### **2.5 Get Pre-trained Model.**
Here, we provide pre-trained weights on SUN dataset, if you do not want to train the models by yourself, you can download it in the following links:
- https://drive.google.com/file/d/1PIwmFLxTXice19-ENFkqg1N25ymZ1s7I/view?usp=drive_link
