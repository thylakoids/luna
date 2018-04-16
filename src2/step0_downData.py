import os
if not os.path.exists('../lunadata/'):
    os.mkdir('../lunadata/')
if not os.path.exists('../lunadata/CSVFILE'):
    os.mkdir('../lunadata/CSVFILE')
if not os.path.exists('../lunadata/CSVFILE/annotations.csv'):
    os.system('wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAANW5Vmi5IClbfh9bboGWwwa/annotations.csv?dl=0 -O ../lunadata/CSVFILE/annotations.csv')
# if not os.path.exists('../lunadata/seg-lungs-LUNA16/'):
# 	os.system('python utils/downGoogle.py 0BxMVN0Vv9w1-WEtYbXc2cXJGa0k ../lunadata/seg-lungs-LUNA16.zip')
# 	os.system('unzip ../lunadata/seg-lungs-LUNA16.zip -d ../lunadata/')
downloadAdrr=['https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAfEPTki7DAzJoB3uAFWRh9a/subset0.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AABJt-eyNWIKcfrWowoeTfzGa/subset1.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAA8PYZxcd7H24SWCN2zSuR6a/subset2.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAB70iYQN8yvTslbbo4-I72pa/subset3.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AABacOTGyiBGXrM012BJZnZLa/subset4.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAs2wbJxbNM44-uafZyoMVca/subset5.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AADuHVxx7RN5Le1n1it_aotka/subset6.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AADsRcbeiDjbJsJ3McD3zYt9a/subset7.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AADE1Z6JUeAQ5Ua6fwETH8yEa/subset8.zip?dl=0',
              'https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AADg2YpNOjYrlqxM_DgqB7T3a/subset9.zip?dl=0',
              ]





for i in range(10):
    if not os.path.exists('../lunadata/subset{}/'.format(i)):
        os.system('wget -O ../lunadata/subset{}.zip {}'.format(i,downloadAdrr[i]))
        os.chdir('../lunadata')
        os.system('7z x subset{}.zip'.format(i))
        os.system('rm subset{}.zip'.format(i))
        os.chdir('../src2')

