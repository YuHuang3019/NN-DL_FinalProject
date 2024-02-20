# Image Caption Generation using CNN and RNN

e4040-2019Fall-Project

* Some very important notes for running the codes:
    1. The accuracy and loss plots are inside the code file and have not been mentioned in the report since we have three models. Kindly refer to the captioning_main.ipynb file to see the accuracy and loss plots.   
    
    2. While running the models, please run the models in order (as the generator is assigned in model 1 cell).  Also, kindly comment the line **vocab_size = 7623** in each model cell Block. This was a small oversight while integrating the codes.
    
    3. The (b) option of our model structure(details shown below) was created and summarized as a figure (shown in the report) but it was not implemented in the final codes because when our group tried to put everything together, we found the encode and decode way for (b) is different to (a) and we encountered some difficulties. 


* This project is based on: 
    1. the paper "Show and Tell: A Neural Image Caption Generator" Vinyals et al., 2015 [https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf]; 
    
    2. the TensorFlow tutorial [https://www.tensorflow.org/tutorials/text/image_captioning];
    
    3. the image captioning introduction [https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8].
   
   
* Original data description: Flickr8k, with 1k training images, 1k devlop images, and 1k test images.
            Each image has 5 captions graded by experts from 0 to 4.


* Our data: because the dataset is huge, we stored our original data, model and weights on the Google Drive, anyone on lionmail should be able to get access to it.     

    1. all original Flickr8k data is stored in this link: https://drive.google.com/drive/folders/1c_NQmQWedz2SP3fh9EoorvsG23Nq8wTw?usp=sharing, 
    
    2. our image features extracted from CNN is saved in this link: https://drive.google.com/drive/folders/1-ECWRNcIegisLmwZczHZ14IQliu_vfoc?usp=sharing
    
    3. our models and weights are saved in this link: https://drive.google.com/drive/folders/1-BghbeP0dYn3ctigZ4fLTUFX1Qi3Jcsj?usp=sharing
    
    4. Kindly refer our way to set the paths, the data could be loaded on Colab. The data file is organised as follows-
       our original data path should be '/content/drive/My Drive/Flickr8k'
       our image features path should be '/content/drive/My Drive/output'
       our models and weights path should be '/content/drive/My Drive/model'


* Our codes: our codes were able to run on Colab.
    1. our preprocessing functions for the text and images are under the folder of *utils* and are called in the main code *captioning_main.ipynb* file.
    
    2. our main code is *captioning_main.ipynb*. 


* Structures:
    1. CNN structure: we directly applied InceptionV3.
    
    2. RNN structure: either LSTM or GRU. (a) We tried to pass both image features and encoded texts to RNN. (b) We also tried to use 2 LSTMs in sequence only apply image information to the first LSTM, and only apply text information to the second LSTM. But this (b) option was not implemented in the final codes because when our group tried to put everything together, we found the encode and decode way for (b) is different to (a) and we encountered some difficulties. 
    
    3. Combined models: please see the report to check the final model structures.


* Our results:
    1. We generally achieved loss of around 4 and accuracy at about 37% ~ 45%.
    
    2. We achieved BLEU-2 score of .23 and BLEU-4 of .14 as compared to 0.3 and 0.17 as the original paper.
