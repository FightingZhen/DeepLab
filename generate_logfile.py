import os

def generate_training_logfile():
    training_dir = "./Data/Eye_picture/Training/"
    test_dir = "./Data/Eye_picture/Test/"

    log_training_dir = "./Data/Eye_picture/train.txt"
    log_test_dir = "./Data/Eye_picture/test.txt"

    training_imagefiles = sorted(os.listdir(training_dir + 'image/'))
    training_maskfiles = sorted(os.listdir(training_dir + 'mask_2class/'))

    test_imagefiles = sorted(os.listdir(test_dir + 'image/'))
    test_maskfiles = sorted(os.listdir(test_dir + 'mask_2class/'))

    log_training = open(log_training_dir, "w")
    log_test = open(log_test_dir, "w")

    for image, mask in zip(training_imagefiles, training_maskfiles):
        log_training.write('image/' + image + ' ' + 'mask_2class/' + mask + '\n')
    for image, mask in zip(test_imagefiles, test_maskfiles):
        log_test.write('image/' + image + ' ' + 'mask_2class/' + mask + '\n')

def generate_practical_logfile(origin_scale=False):
    if not origin_scale:
        image_dir = "./Data/Practical_Eye_image"

        log_training_dir = "./Data/Practical_Eye_image/test.txt"
    else:
        image_dir = "./Data/Practical_Origin_Eye_image"

        log_training_dir = "./Data/Practical_Origin_Eye_image/test.txt"

    training_imagefiles = sorted(os.listdir(image_dir))

    log_training = open(log_training_dir, "w")

    for image in training_imagefiles:
        log_training.write(image + '\n')

if __name__ == '__main__':
    generate_training_logfile()
    generate_practical_logfile(origin_scale=True)
    generate_practical_logfile(origin_scale=False)