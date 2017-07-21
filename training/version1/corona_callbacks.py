import keras
import time

class CoronaCallbacks(keras.callbacks.Callback):

    def __init__(self, filepath, network_arguments):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = filepath + "/" + timestr
        self.argument_string = arguments_to_csv_row(network_arguments)
        with open(self.filepath, "wb") as out:
            csv_header_string =  arguments_to_csv_header(network_arguments)
            out.write(csv_header_string)
            out.write("\n")
        
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_train_end(self, logs={}):
        
        with open(self.filepath, "ab") as out:
            for loss in self.losses:
                out.write(str(loss[0]))
                out.write(",")
                out.write(str(loss[1]))
                out.write(self.argument_string)
                out.write("\n")
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append([logs.get('val_loss'), logs.get('loss')])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def arguments_to_filename(args):
    """
    Take the arguments object from argparse and turn it into a filename.
    """
    arg_dict = args.__dict__
    arg_dict.pop("ignore", None)
    l = list(arg_dict)
    l = sorted(l)
    filename = "search"
    for arg in l:
        filename += "--" + arg + "-" + str(arg_dict[arg])
    return filename + ".csv"

def arguments_to_csv_header(args):
    """
    Take the arguments from argparse and produce a string for the header value.
    This will have the form "epoch number,training score,test score,layer1 parameter1,..."
    """
    header = "validation loss, training loss"
    arg_dict = args.__dict__
    arg_dict.pop("ignore", None)
    l = list(arg_dict)
    l = sorted(l)
    for arg in l:
        header += "," + arg
    return header
    
def arguments_to_csv_row(args):
    """
    Take the arguments from the argparse object and turn it into the string
    that will be added to every row. This is useful since it will be easier
    to load into analytical programming environments like R.
    """
    row = ""
    arg_dict = args.__dict__
    arg_dict.pop("ignore", None)
    l = list(arg_dict)
    l = sorted(l)
    for arg in l:
        row += str(arg_dict[arg]) + ","
    return row
