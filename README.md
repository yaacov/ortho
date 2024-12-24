## HOCR

### OCR pipline:

1. load and clean image : <image name>_bw.jpg
2. estemate char height
3. get column masks     : <image name>_columns.jpg
4. get line masks       : <image name>_lines.jpg
5. get line images      : <image name>_c<number>_l<number>.jpg
6. ocr lines            : <image name>.hocr

### AI piplines:

#### Train font_clasifier model:

1. load and clean image : <image name>_bw.jpg
2. estemate char height
3. get gliphs
4. train font classifier on labeled gliphs 

#### Train OCR model:

1. train on line images with known texts

## scripts:

load an image and convert it to b/w image + extract fonts for font_clasifier training

```bash
python image_proccessing.py
```

train font_clasifier

```bash
python glyph_classifier.py --train --resume

python glyph_classifier.py --predict chars --max_files 10000
```


### TensorBoard Usage:

To visualize the training progress, use TensorBoard:

```bash
tensorboard --logdir=runs/font_nikud_classifier
```

This will start a local server. You can access it by opening a web browser and going to: http://localhost:6006
You'll be able to view various metrics such as training loss over time, which helps monitor the training process.
