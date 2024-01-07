import sys
import os
#from textatistic import Textatistic
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
#import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

"""
Main File for the evaluation
Arguments:
1 path to the model output dir to be evaluated
2 dev or test
3 path to annotation file
"""

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='../assets/coco'
dataType='merged2014'
algName = 'fakecap'
#annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
annFile = sys.argv[3]
subtypes=['results', 'evalImgs', 'eval']
'''
[resFile, evalImgsFile, evalFile]= \
['./results/captions_%s_%s_%s.json'%(dataType,algName,subtype) for subtype in subtypes]
'''
#resFile = "../train_data/ms_coco/ms_coco_model/model_1/output_devformated.json"
resDir = sys.argv[1]
resFile = resDir+"/output_"+sys.argv[2]+"formated.json"
# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)

# read the results of human evaluation and print average scores
humanResDir = resDir+"/humanEvaluations/"
all_files = os.listdir(humanResDir)
naturalness_scores = []
quality_scores = []
info_scores = []
for imgFile in all_files:
    with open(humanResDir+imgFile, "r") as f:
        evalparsed = json.load(f)
        naturalness_scores += evalparsed["naturalness"]
        quality_scores += evalparsed["quality"]
        info_scores += evalparsed["informativeness"]
naturalness_scores = [float(x) for x in naturalness_scores]
quality_scores = [float(x) for x in quality_scores]
info_scores = [float(x) for x in info_scores]
print("Average human evaluation score for naturalness: ", sum(naturalness_scores)/len(naturalness_scores))
print("Average human evaluation score for quality: ", sum(quality_scores)/len(quality_scores))
print("Average human evaluation score for informativeness: ", sum(info_scores)/len(info_scores))

# compute flesch score
'''
sentences = u''
for image in cocoRes.dataset["annotations"]:
	sentences += image["caption"] + '. '
flesch = Textatistic(sentences).flesch_score
print ("Flesch score: ",flesch)
'''

