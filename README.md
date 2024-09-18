# AlexNet and Image Classification

<p align="justify">
<strong>AlexNet</strong> is a pioneering deep learning architecture specifically designed for image classification tasks. Introduced by Alex Krizhevsky and his team in 2012, AlexNet significantly advanced the field of computer vision by achieving remarkable performance on the ImageNet dataset, a large-scale image recognition benchmark. The architecture features several convolutional layers, max-pooling layers, and fully connected layers, allowing it to effectively learn and recognize complex patterns in images.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_AlexNet_ImageClassification_uddrlyxa.git
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_alexnet_train_sxbavvhx.py --train_dir ../dataset --num_epochs 100 --batch_size 32 --model_path aicandy_model_out_uoebddte/aicandy_model_pth_luveqrpt.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_alexnet_test_ovpridiv.py --image_path ../image_test.jpg --model_path aicandy_model_out_uoebddte/aicandy_model_pth_luveqrpt.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_alexnet_convert_onnx_rvecsgnu.py --model_path aicandy_model_out_uoebddte/aicandy_model_pth_luveqrpt.pth --onnx_path aicandy_model_out_uoebddte/aicandy_model_onnx_sriklays.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-alexnet-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




