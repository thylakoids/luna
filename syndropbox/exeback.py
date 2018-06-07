import os 
import sys
print os.getcwd(),'$'
print time.strftime("%Y-%m-%d %H:%M:%S")
print '*'*24
print '\n'

#copy 
os.system('cp ~/Dropbox/loadDataTrain.py src2/loadDataTrain.py')
#run
os.system('nohup python src2/loadDataTrain.py > sgd.out 2>&1 &')


#monitor
os.system('nvidia-smi')
print '-'*50
print '\n'
