This is the official PyTorch implementation of Stitchable Neural Networks.

By [MohammadMahdi Azizi](https://github.com/mhmdmhdiazizi), [Setareh Abhari](https://github.com/setarehabhari), Dr. Hedieh Sajedi.

# Stitched MedViT

In this research, we were able to demonstrate that for a dataset of Optical Coherence Tomography images, even on different datasets, utilizing a stitchable neural network as an architecture search method for efﬁcient neural networks based on pre-trained medical vision transformers, in a few epochs of training can result in an accurate model in classifying images from our original dataset.



# Pre-Training
This folder consists of the MedVit project which was forked from github. In The given files we have altered the Model to take our dataset as an input, train the MedVit model on them, and then return the pre-trained model checkpoint as a result. The file can be executed using the following command:

```
python /pre-training/main.py \
--num_epochs 75 \
--data_file_train TRAIN_INFO_FILE \
--data_file_val VAL_INFO_FILE \
--data_file_test TEST_INFO_FILE  \
--data_root ./NEH_UT_2021RetinalOCTDataset \
--class_name_column class-id --image_column file-name \
--n_classes 3 --task single-label --batch_size 32 \
--model_flag MedViT_micro --data_flag NEH \
--run EXPERIMETN_NAME

```

# Stitching MedViTs
This folder consists of the code for the stitchable MedViT model. We have used the codes for Stitchable Neural Network from Git Hub and altered it to use MedViT as its anchors and to run on our given dataset. To run the altered Model you can Use the following command:

```
python /stitchingmedvits/main.py \
--output-dir DIRECTORY_FOR_OUTPUTS \
--epochs 100 --batch-size 32  \
--lr 5e-5 --warmup-lr 1e-7 --min-lr 1e-6  \
--data_root ./NEH_UT_2021RetinalOCTDataset \
--data_file_train TRAIN_INFO_FILE \
--data_file_val VAL_INFO_FILE \
--image_column file-name --class_name_column class-id \
--nb_classes 3 --distillation-type  none \
--teacher-path PATH_TO_THEtEACHER_MODEL \
--teacher-model MedViT_micro 
```


# Acknowledgement

This implementation is built upon [MedViT](https://github.com/Omid-Nejati/MedViT) and [Stitchable Neural Networks](https://github.com/ziplab/SN-Net/). We thank the authors for their released code.


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](/LICENSE) file.
