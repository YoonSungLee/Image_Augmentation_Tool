# Image_Augmentation_Tool

This repo is Image Augmentation Tool when you need to increase your custom dataset for some reason.
If you download this repo, please follow the procedure below.
<br><br><br>
### 1. Imgaug Install
make a virtual environment, and execute the code below.<br>
`pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely imgaug pascal_voc_writer`
<br><br><br>
### 2. Input Your Custom Dataset In 'before_dataset'
I created a 'before_dataset' folder for you to understand. Input your custom datset in this folder. Remember to put the png files and xml files under the same name, as shown in the example.
<br><br><br>
### 3. Set The Argumentation Way You Want.
In 'augmentation.py' file, you are able to see the code below. Set the argumentation way you want. If you want to know the mean of 'parameter', please visit the site 'imgaug' below.
```python
    seq = iaa.Sequential([
        # iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            # translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7),
            # rotate=45
        )
    ])
```
<br><br><br>
### 4. Run 'argumentation.py'
If you run the code below, you are able to get your argumented custom dataset in 'after_dataset' folder.<br>
`python argumentation.py`
<br><br><br>
### Plus. Run 'argumentation_preview.py'
I created a 'argumentation_preview.py' file. If you want to preview your argumented custom dataset, please run the code below. Remember that this file show you argumented custom dataset, not to save.
<br><br><br>
### Reference
I reveal that I have quoted two sites below. For more specific information, visit the site below.<br>
<br>
[Image Augmentation Blog](https://junyoung-jamong.github.io/machine/learning/2019/01/23/%EB%B0%94%EC%9A%B4%EB%94%A9%EB%B0%95%EC%8A%A4%EB%A5%BC-%ED%8F%AC%ED%95%A8%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%A6%9D%ED%8F%AD%EC%8B%9C%ED%82%A4%EA%B8%B0-with-imgaug.html)<br>
[Imgaug Site](https://github.com/aleju/imgaug)
