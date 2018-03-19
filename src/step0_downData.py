import os

if not os.path.exists('../lunadata/'):
	os.mkdir('../lunadata/')
if not os.path.exists('../lunadata/CSVFILE'):
	os.mkdir('../lunadata/CSVFILE')
if not os.path.exists('../lunadata/CSVFILE/annotations.csv'):
	os.system('wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAANW5Vmi5IClbfh9bboGWwwa/annotations.csv?dl=0 -O ../lunadata/CSVFILE/annotations.csv')
if not os.path.exists('../lunadata/seg-lungs-LUNA16/'):
	os.system('python utils/downGoogle.py 0BxMVN0Vv9w1-WEtYbXc2cXJGa0k ../lunadata/seg-lungs-LUNA16.zip')
	os.system('unzip ../lunadata/seg-lungs-LUNA16.zip -d ../lunadata/')
if not os.path.exists('../lunadata/subset0/'):
	os.system('wget -O ../lunadata/subset0.zip https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAfEPTki7DAzJoB3uAFWRh9a/subset0.zip?dl=0')
	os.system('cd ../lunadata')
	os.system('7z x subset0.zip')
	os.system('cd ../src')
