from utils.pathname import correct_path
class Config:
    FOLDERS = ['rawdata']
    TESTING = False
    ANNOTATION_PATH = correct_path('../lunadata/CSVFILE/annotations.csv')
    STEPS = 5
    EPOCHS = 3
class TestingConfig(Config):
    TESTING = True
class ProductionConfig(Config):
    FOLDERS = ['subset{}'.format(i) for i in range(10)]
    STEPS = 200
    EPOCHS = 30
class DevelopmentConfig(Config):
    pass
config = {
    'Testing': TestingConfig,
    'Production':ProductionConfig,
    'Development':DevelopmentConfig
}

conf= config['Testing']
