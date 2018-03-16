import os

# if not os.path.exists('../lunadata/CSVFILE'):
# 	os.mkdir('../lunadata')
# 	os.mkdir('../lunadata/CSVFILE')
if not os.path.exists('../lunadata/CSVFILE/annotations.csv'):
	os.system('wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAANW5Vmi5IClbfh9bboGWwwa/annotations.csv?dl=0 ../lunadata/CSVFILE/annotations.csv')