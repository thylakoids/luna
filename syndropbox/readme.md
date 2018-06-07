# crossentropy (0.544 on test)
    model1
    bachsize = 32 (16 positive)
    20 epochs
    lr = 1e-5

# sdg
    model2 (not good)

# e6
    model3
    reduce bachsize to 16
    reduce lr to 1e-6
    30 epochs

# Unet4.out
    model4 retrain on model1
    reduce bachsize to 16
    reduce lr to 1e-6
    10 epochs

    model5 retrain on model4

    model6 retrain on model5

# baidu yun
链接: https://pan.baidu.com/s/16bU7NxBsFTAWJPYPsocPMA 密码: e63s


