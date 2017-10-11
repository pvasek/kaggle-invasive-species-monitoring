import os

def get_train_map(file_name):
    """
    Reads the csv file and returns array of numbers
    0 - non invasive
    1 - invasive
    """
    with open(file_name) as f:
        lines = f.readlines()
        dict = {}
        for x in lines[1:]:
            items = x.strip().split(',')
            dict[items[0] + '.jpg'] = 1 if items[1] == '1' else 0
        return dict

def align_train_files(directory, labels):
    invasive_dir = os.path.join(directory, '1_invasive')
    noninvasive_dir = os.path.join(directory, '0_noninvasive')

    for i in [invasive_dir, noninvasive_dir]:
        if os.path.exists(i) == False:
            os.makedirs(i)

    map = get_train_map(labels)
    dir_items = os.scandir(directory)
    files = [file for file in dir_items if file.is_file()]
    for file in files:
        file_name = file.name
        if file_name in map:
            target_dir = os.path.join(invasive_dir if map[file_name] else noninvasive_dir, file_name)
            os.rename(file.path, target_dir)

def pickup_valiation_data(train_directory, validation_directory, amount=0.2):
    print('validation_directory: {0}'.format(validation_directory))
    if os.path.exists(validation_directory):
        print('validation directory already exists')
        return

    sub_directories = [file for file in os.scandir(train_directory) if file.is_dir()]
    print('class subdirectories: {1}'.format(validation_directory, [i.name for i in sub_directories]))

    for i in sub_directories:
        if os.path.exists(i.path) == False:
            os.makedirs(os.path.join(validation_directory, i.name))

    for sub_directory in sub_directories:
        files = [file for file in os.scandir(sub_directory.path) if file.is_file()]
        random.shuffle(files)
        count = int(len(files) * amount)
        print('total number of files: {0}, pickuped for validation: {1}'.format(len(files), count))
        files_for_validation = files[0:count]
        for i in files_for_validation:
            destination = os.path.join(validation_directory, sub_directory.name, i.name)
            os.rename(i.path, destination)
