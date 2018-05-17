from utils.pathname import correct_path
class Config:
    FOLDERS = ['rawdata']
    TESTING = False
    ANNOTATION_PATH = correct_path('../lunadata/CSVFILE/annotations.csv')
    STEPS = 5
    EPOCHS = 3
    BATCHSIZE = 8
class TestingConfig(Config):
    TESTING = True
class ProductionConfig(Config):
    FOLDERS = ['subset{}'.format(i) for i in range(10)]

    BATCHSIZE = 32
    STEPS = int(500/BATCHSIZE)# 4 steps one epoch
    EPOCHS = 20
class DevelopmentConfig(Config):
    pass
config = {
    'Testing': TestingConfig,
    'Production':ProductionConfig,
    'Development':DevelopmentConfig
}

conf= config['Development']
