import keras.utils.data_utils
import random

class GeneratorSequence(keras.utils.data_utils.Sequence):
    """
    The sequence class ensures the data returned to the training/validation processes is
    in a reliable sequence and can be run on multiple processes. Without this class,
    you will not be able to prepare multiple sets of data for the GPU in parallel.
    """

    def __init__(self, dataset_model_class=None, batch_size=-1, multiprocess=True,
                 dataset_model_params={side_channels=-1, aia_image_count=-1, dependent_variable=None}):
        """Copy the current dataset model so it will sit in this separate
        process. The separate process will then pass ready objects
        between processes by pickling.
        @param dataset_model {DatasetModel|None} a dataset model. Multiprocessing batches should
        pass None into this parameter since the process will instantiate the object.
        """
        self.dataset_model_class = dataset_model_class
        self.batch_size = batch_size
        self.dataset_model_params = dataset_model_params

    def __len__(self):
        """@return the number of samples included in each batch"""
        return self.batch_size

    def __getitem__(self, batch_idx, dataset_model_class=None, dataset_model_params=None):
        """Get a batch of samples. If the batch size will take the sampling
        past the current list of files, then the files list will be
        shuffled and the sampling will continue from the beginning.
        """
        dataset_model = dataset_model_class(dataset_model_params)
        files = self.dataset_model.train_files
        random.seed(batch_idx) # Standardize the random number generator to consistent shuffles
        random.shuffle(files)
        data_x = []
        data_y = []
        for count in range(0, self.batch_size):
            file_index = self._get_file_index(batch_idx, count)
            f = files[file_index]
            sample = dataset_model._get_x_data(f,
                                               aia_image_count=dataset_model.aia_image_count,
                                               training=True)
            dataset_model._sample_append(data_x, sample)
            data_y.append(dataset_model._get_y(f))
        finalized = dataset_model._finalize_dataset(data_x, data_y)
        return finalized

    def _get_file_index(self, batch_idx, idx):
        """Return the index of the current item that should be returned."""
        # todo: the shuffling is wonky. It is currently just generating a random sample
        file_index = (batch_idx * self.batch_size + idx) % file_count
        return file_index
