from utils.pathname import correct_path
class Config:
    FOLDERS = ['rawdata']
    TESTING = False
    ANNOTATION_PATH = correct_path('../lunadata/CSVFILE/annotations.csv')
class TestingConfig(Config):
    TESTING = True
class ProductionConfig(Config):
    FOLDERS = ['subset{}'.format(i) for i in range(10)]
class DevelopmentConfig(Config):
    pass
config = {
    'Testing': TestingConfig,
    'Production':ProductionConfig,
    'Development':DevelopmentConfig
}

conf= config['Production']
