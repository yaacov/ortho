## ORTHO

Old Research Text Hebrew OCR

### Usage:

```sh
# Install required python libraries
make install

# Run a script to test image cleaning
python scripts/prepare_images.py

# Run a script to test glyph cleaning
python scripts/prepare_glyphs.py

# Run a script to test line cleanning
python scripts/prepare_lines.py

# Run a script to test glyph classifier
# ! Requires clean glyphs, use prepare_glyphs.py to prepare the glyphs
python scripts/test_glyph_classifier.py

# Optional, after running the test glyph classifier,
# copy a random selection of classified glyphs for manual review
python scripts/random_file_copier.py

# Run a script to train glyph classifier
# ! Requires classified glyphs as labeled examples,
#   use test_glyph_classifier.py to label glyphs for training
python scripts/train_glyph_classifier.py

# Run a script to test:
# - find glyphs in page
# - clasify text glyphs in page 
python scripts/guess_image_glyphs.py 

# Run a script to test:
# - find glyphs in page
# - clasify text paragraphs in page 
python scripts/guess_image_paragraphs.py

# Run a script to test:
# - find glyphs in page
# - clasify text lines in page
python scripts/guess_image_lines.py

# Run a script to create masks:
# - resize images for uniform font size
# - create line detection masks
python scripts/prepare_line_masks.py

# Test masks by overlaying masks on input images
python scripts/overlay_masks.py
```

### TensorBoard Usage:

To visualize the training progress, use TensorBoard:

```bash
tensorboard --logdir=runs/font_nikud_classifier
```

This will start a local server. You can access it by opening a web browser and going to: http://localhost:6006
You'll be able to view various metrics such as training loss over time, which helps monitor the training process.
