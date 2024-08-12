#[ECCV24] Bones Can't Be Triangles: Accurate and Efficient Vertebrae Keypoint Estimation through Collaborative Error Revision 

[[Project Page]](https://ts-kim.github.io/KeyBot/)
## Introduction

This is the official PyTorch implementation of "Bones Can't Be Triangles: Accurate and Efficient Vertebrae Keypoint Estimation through Collaborative Error Revision (ECCV 2024)."


## Environment Setup

This code was developed using Python 3.8 on an Ubuntu 18.04 system.

## Quick start

### Installation

1. **Install Required Packages**:
   Use `pip` to install the necessary Python packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   
   - **Obtain the dataset**: The AASCE dataset can be requested from [this link](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images).
   
   - **Organize the dataset**: Move the downloaded dataset to the following directory structure:
     ```
     codes/preprocess_data/AASCE_rawdata/boostnet_labeldata
     ```
   
   - **Run preprocessing**: Navigate to the preprocessing code directory and execute the preprocessing script:
     ```bash
     cd codes/preprocess_data/
     python preprocess_data.py
     cd ..
     ```
   
### How to use


1. **Training Your Own Model**:
   
   To train your model, execute the following command:
   ```
   bash train_interactive_keypoint_model.sh
   python train_AASCE.py
   ```
2. **Inference**:
   
   Once the data is prepared, run the following command to perform inference with the pre-trained model:
   ```
   python evaluate_AASCE.py
   ```


## Citation

If you find this work or code is helpful in your research, please cite:
```
@inproceedings{kim2024Bones,
  title={Bones Can't Be Triangles: Accurate and Efficient Vertebrae Keypoint Estimation through Collaborative Error Revision},
  author={Kim, Jinhee and Kim, Taesung and Choo, Jaegul},
  booktitle={European Conference on Computer Vision},
  year={2024},
}
```