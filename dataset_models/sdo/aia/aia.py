import yaml
import os

class AIA:
    """
    A class for managing the download
    and interface of the AIA data.
    """

    def __init__(self, dependent_variable="flux delta"):
        """
        Get a directory listing of the AIA data and load all the filenames
        into memory. We will loop over these filenames while training or
        evaluating the network.
        @param dependent_variable {enum} The valid values for this
        enumerated type are 'flux delta', which indicates we are concerned
        with predicting the change in x-ray flux through time, or
        'forecast' which is concerned with predicting the total x-ray flux
        output at the next time step.
        """

        self.dependent_variable = dependent_variable

        with open("config.yml", "r") as config_file:
            self.config = yaml.load(config_file)
        assert(self.is_downloaded())
        train_files = os.listdir(self.config["aia_path"] + "training")
        validation_files = os.listdir(self.config["aia_path"] + "validation")

        self.y_dict = {}
        with open(self.config["aia_path"] + "y/Y_GOES_XRAY_201401.csv", "rb") as f:
            for line in f:
                split_y = line.split(",")
                self.y_dict[split_y[0]] = float(split_y[1])

    def is_downloaded(self):
        """
        Determine whether the AIA dataset has been downloaded.
        """
        if not os.path.isdir(self.config["aia_path"]):
            print("WARNING: the data directory specified in config.yml does not exist")
            return False
        if not os.path.isdir(self.config["aia_path"] + "validation"):
            print("WARNING: you have no validation folder")
            print("place these data into " + self.config["aia_path"] + "validation")
            return False
        if not os.path.isdir(self.config["aia_path"] + "training"):
            print("WARNING: you have no training folder")
            print("place these data into " + self.config["aia_path"] + "training")
            return False
        if not os.path.isdir(self.config["aia_path"] + "y"):
            print("WARNING: you have no dependent variable folder")
            print("place these data into " + self.config["aia_path"] + "y")
            return False
        if not os.path.isfile(self.config["aia_path"] + "y/Y_GOES_XRAY_201401.csv"):
            print("WARNING: you have no results dataset")
            print("place these data into " + self.config["aia_path"] + "y")
            return False
        if not os.path.isfile(self.config["aia_path"] + "training/20140121_1400_AIA_08_1024_1024.dat"):
            print("WARNING: you have no independent variable training dataset")
            print("place these data into " + self.config["aia_path"] + "training")
            return False
        if not os.path.isfile(self.config["aia_path"] + "validation/20140105_1724_AIA_08_1024_1024.dat"):
            print("WARNING: you have no independent variable validation dataset")
            print("place these data into " + self.config["aia_path"] + "validation")
            return False
        return True

    def get_y(self, filename):
        """
        Get the true forecast result for the current filename.
        """
        split_filename = filename.split("_")
        k = split_filename[0] + "_" + split_filename[1]
        future = self.y_dict[k]

        if self.dependent_variable == "flux delta":
            current = self.get_prior_y(filename)
            return abs(future - current)
        elif self.dependent_variable == "forecast":
            return future
        else:
            assert False # There are currently no other valid dependent variables
            return None

    def get_prior_y(self, filename):
        """
        Get the y value for the prior time step. This will
        generally be used so we can capture the delta in the
        prediction value.
        """
        f = filename.split("_")
        datetime_format = '%Y%m%d_%H%M'
        datetime_object = datetime.strptime(f[0]+"_"+f[1], datetime_format)
        td = timedelta(minutes=-12)
        prior_datetime_object = datetime_object + td
        prior_datetime_string = datetime.strftime(prior_datetime_object, datetime_format)
        return self.y_dict[prior_datetime_string]

    #  Dictionary caching filenames to their normalized in-memory result
    cache = {}
    def generator(self, training=True):
        """
        Generate samples
        """

        def available_cache(training):
            """
            If there is enough main memory, add the object to the cache.
            Always add the object to the cache if it is in the validation
            set.
            """
            vm = psutil.virtual_memory()
            return vm.percent < 75 or not training

        if training:
            files = train_files
        else:
            files = test_files

        x_mean_vector = [2.2832, 10.6801, 226.4312, 332.5245, 174.1384, 27.1904, 4.7161, 67.1239]
        x_standard_deviation_vector = [12.3858, 26.1799, 321.5300, 475.9188, 289.4842, 42.3820, 10.3813, 72.7348]

        data_x = []
        data_y = []
        i = 0
        while 1:

            # The current file
            f = files[i]

            # Get the sample from the cache or load it from disk
            if f in cache:
                data_x_sample = cache[f][0]
                data_y_sample = cache[f][1]
            else:
                data_x_sample = np.load(data_directory + f)
                data_x_sample = ((data_x_sample.astype('float32').reshape((input_width*input_height, input_channels)) - x_mean_vector) / x_standard_deviation_vector).reshape((input_width, input_height, input_channels)) # Standardize to [-1,1]
                data_y_sample = get_y(f)

                if available_cache(training):
                    cache[f] = [data_x_sample, data_y_sample]

            data_x.append(data_x_sample)
            data_y.append(data_y_sample)

            if i == len(files):
                i = 0
                random.shuffle(files)

            if samples_per_step == len(data_x):
                ret_x = np.reshape(data_x, (len(data_x), input_width, input_height, input_channels))
                ret_y = np.reshape(data_y, (len(data_y)))
                yield (ret_x, ret_y)
                data_x = []
                data_y = []
