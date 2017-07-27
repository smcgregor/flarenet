import os

def change_directory_to_root():
    """
    Ensure the script is executing from the root of the project.
    """
    abspath = os.path.abspath(__file__)
    head, tail = os.path.split(abspath)
    os.chdir(head + "/..")

def construct_training_file_structure(training_directory_path, run_name):
    """
    This function will construct the output folders expected for the
    training process.
    @param training_directory_path {string} Where relative to the root
    we will construct the path.
    @param run_name {string} The string identifying this particular run.
    This is necessary since the architectures script may execute several different
    architectures.
    """
    if not os.path.exists(training_directory_path):
        os.makedirs(training_directory_path)
    os.makedirs(training_directory_path + run_name)
    training_directory_path = training_directory_path + run_name
    directories = ["epochs", "performance", "maps", "features", "embeddings"]
    for directory in directories:
        print training_directory_path + directory
        if not os.path.exists(training_directory_path + directory):
            os.makedirs(training_directory_path + directory)
