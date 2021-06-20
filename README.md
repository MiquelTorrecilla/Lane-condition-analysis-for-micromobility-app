# Lane-condition-analysis-for-micromobility-applications

Due to the increase in population in large urban areas, mobility in large cities has become increasingly problematic and difficult. With the desire to find solutions to this situation, the use of micro-mobility vehicles has become popular.

In this sense the solution presented proposes to make use of Deep Learning and Computer Vision techniques in order to identify, automatically, possible defects where micro-mobility vehicles circulate, in order to alert authorities and other users and improve user safety.

Three solutions were devised to determine which one best suited the needs of the project. The first two to test the two models separately and the third one with the intention of combining them, with ensemble learning techniques, to try to improve the results obtained.


## Main
*We recommend to use it in google colab*

This file have all the code to train, modify the configs, show some results, make use of the Non-maxima suppresion and compute mAP for the two models.
For the training part we recomend to use it on the serverof TSC.

## Training in the server

**first you need to create a virtualenv, execute all this orders:**

virtualenv --python=/usr/bin/python3.9 ~/venv/mm2

source ~/venv/mm2/bin/activate

srun --mem 6G --gres=gpu:1 --time=10:00:00 pip install torch torchvision

srun --mem 6G --gres=gpu:1 --time=10:00:00 pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/102/1.8.1/index.html

cd workspace

git clone https://github.com/open-mmlab/mmdetection.git

srun --mem 6G --gres=gpu:1 --time=10:00:00 pip install -r requirements/build.txt

srun --mem 6G --gres=gpu:1 --time=10:00:00 pip install -v -e .

srun --mem 6G --gres=gpu:1 --time=10:00:00 python test.py

**Then you need to execute this orders so you can train the detectors:**

*We have created a virtualenv named mm3.*

module load cudnn/7.4 cuda/10.2

source ~/venv/mm3/bin/activate

cd workspace

cd mmdetection/

srun --mem 6G --gres=gpu:1 --time=10:00:00 python FasterRCNN.py

srun --mem 6G --gres=gpu:1 --time=10:00:00 python CascadeRCNN.py

In the directory CFG we have the configs for the two models.

## Results
*we recommend to use it in google colab*

Compute Results is the code you have to execute to obtain the F1SCORE for the two models per classes.
Yo have to upload the bboxes obtain with the two models. The bboxes are in the Results directory.




