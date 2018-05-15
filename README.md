# luna
luna16 
<!-- MarkdownTOC -->

1. [step0_downData](#step0_downdata)
1. [step1_preprocess](#step1_preprocess)
1. [videowjpg jpg2video](#videowjpg-jpg2video)

<!-- /MarkdownTOC -->

## step0_downData
## step1_preprocess
**1.0** load ct scan and generate Lung segmentation
```python 
for file in datafolder:
	read image
	Segmentation of Lungs
	normalization and zero centen
	save Segmentation of Lungs
```
**1.1** generate **2d** train and test data set
```python
for 3DsegmentedLungs:
	create nodual mask using annotations
	load 3D Segmentation of Lungs
	resample 3DsegmentedLungs and masks
	generate 2D slices
```
## [videowjpg jpg2video](http://www.scikit-video.org/stable/index.html)