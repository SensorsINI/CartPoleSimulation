""" Shared stuff between producer and consumer
 Author: Tobi Delbruck
 Source: https://github.com/SensorsINI/joker-network
 """
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import dictdiffer as dictdiffer
import ruamel.yaml as yaml # correctly supports scientific notation for numbers, see https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# import yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
# from generallibrary import print_link, print_link_to_obj # for logging links to source code in logging output for pycharm clicking, see https://stackoverflow.com/questions/26300594/print-code-link-into-pycharms-console
from munch import Munch, DefaultMunch
from numba import jit

from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import ComputationLibrary, TensorType

log=get_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # all TF messages

import atexit
# https://stackoverflow.com/questions/35851281/python-finding-the-users-downloads-folder
import os

import numpy as np
import tensorflow as tf
from engineering_notation import EngNumber as eng  # only from pip
from matplotlib import pyplot as plt
from numpy.random import SFC64, Generator

if os.name == 'nt':
    import ctypes
    from ctypes import windll, wintypes
    from uuid import UUID

    # ctypes GUID copied from MSDN sample code
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8)
        ]

        def __init__(self, uuidstr):
            uuid = UUID(uuidstr)
            ctypes.Structure.__init__(self)
            self.Data1, self.Data2, self.Data3, \
            self.Data4[0], self.Data4[1], rest = uuid.fields
            for i in range(2, 8):
                self.Data4[i] = rest>>(8-i-1)*8 & 0xff

    SHGetKnownFolderPath = windll.shell32.SHGetKnownFolderPath
    SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(GUID), wintypes.DWORD,
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)
    ]

    def _get_known_folder_path(uuidstr):
        pathptr = ctypes.c_wchar_p()
        guid = GUID(uuidstr)
        if SHGetKnownFolderPath(ctypes.byref(guid), 0, 0, ctypes.byref(pathptr)):
            raise ctypes.WinError()
        return pathptr.value

    FOLDERID_Download = '{374DE290-123F-4565-9164-39C4925E467B}'

    def get_download_folder():
        return _get_known_folder_path(FOLDERID_Download)
else:
    def get_download_folder():
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")

LOGGING_LEVEL = logging.INFO
PORT = 12000  # UDP port used to send frames from producer to consumer
IMSIZE = 224  # input image size, must match model
UDP_BUFFER_SIZE = int(math.pow(2, math.ceil(math.log(IMSIZE * IMSIZE + 1000) / math.log(2))))

EVENT_COUNT_PER_FRAME = 2300  # events per frame
EVENT_COUNT_CLIP_VALUE = 3  # full count value for colleting histograms of DVS events
SHOW_DVS_OUTPUT = True # producer shows the accumulated DVS frames as aid for focus and alignment
MIN_PRODUCER_FRAME_INTERVAL_MS=7.0 # inference takes about 3ms and normalization takes 1ms, hence at least 2ms
        # limit rate that we send frames to about what the GPU can manage for inference time
        # after we collect sufficient events, we don't bother to normalize and send them unless this time has
        # passed since last frame was sent. That way, we make sure not to flood the consumer
MAX_SHOWN_DVS_FRAME_RATE_HZ=15 # limits cv2 rendering of DVS frames to reduce loop latency for the producer
FINGER_OUT_TIME_S = 2  # time to hold out finger when joker is detected
ROOT_DATA_FOLDER= os.path.join(get_download_folder(),'trixsyDataset') # does not properly find the Downloads folder under Windows if not on same disk as Windows

DATA_FOLDER = os.path.join(ROOT_DATA_FOLDER,'data') #/home/tobi/Downloads/trixsyDataset/data' #'data'  # new samples stored here
NUM_NON_JOKER_IMAGES_TO_SAVE_PER_JOKER = 3 # when joker detected by consumer, this many random previous nonjoker frames are also saved
JOKERS_FOLDER = DATA_FOLDER + '/jokers'  # where samples are saved during runtime of consumer
NONJOKERS_FOLDER = DATA_FOLDER + '/nonjokers'
SERIAL_PORT = "/dev/ttyUSB0"  # port to talk to arduino finger controller

LOG_DIR='logs'
SRC_DATA_FOLDER = os.path.join(ROOT_DATA_FOLDER,'source_data') #'/home/tobi/Downloads/trixsyDataset/source_data'
TRAIN_DATA_FOLDER=os.path.join(ROOT_DATA_FOLDER,'training_dataset') #'/home/tobi/Downloads/trixsyDataset/training_dataset' # the actual training data that is produced by split from dataset_utils/make_train_valid_test()


MODEL_DIR='models' # where models stored
JOKER_NET_BASE_NAME='joker_net' # base name
USE_TFLITE = True  # set true to use TFLITE model, false to use full TF model for inference
TFLITE_FILE_NAME=JOKER_NET_BASE_NAME+'.tflite' # tflite model is stored in same folder as full-blown TF2 model
CLASS_DICT={'nonjoker':1, 'joker':2} # class1 and class2 for classifier
JOKER_DETECT_THRESHOLD_SCORE=.95 # minimum 'probability' threshold on joker output of CNN to trigger detection

import signal


def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=30):
    """ get input with timeout
    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable
    :returns: the input
    :raises: TimeoutError if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout) # produce SIGALRM in `timeout` seconds
    try:
        time.sleep(.5) # get input to be printed after logging
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0) # cancel alarm

def yes_or_no(question, default='y', timeout=None):
    """ Get y/n answer with default choice and optional timeout from terminal input.

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n'
    :param timeout: the timeout in seconds, default is None

    :returns: True or False
    """
    if default is not None and (default!='y' and default!='n'):
        log.error(f'bad option for default: {default}')
        quit(1)
    y='Y' if default=='y' else 'y'
    n='N' if default=='n' else 'n'
    while "the answer is invalid":
        try:
            to_str='' if timeout is None or os.name=='nt' else f'(Timeout {default} in {timeout}s)'
            if os.name=='nt':
                log.warning('cannot use timeout signal on windows')
                time.sleep(.1) # make the warning come out first
                reply=str(input(f'{question} {to_str} ({y}/{n}): ')).lower().strip()
            else:
                reply = str(input_with_timeout(f'{question} {to_str} ({y}/{n}): ',timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply=''
        if len(reply)==0 or reply=='':
            return True if default=='y' else False
        elif reply[0].lower() == 'y':
            return True
        if reply[0].lower() == 'n':
            return False




def create_rng(id: str, seed: str, use_tf: bool=False):
    if seed == None:
        log.info(f"{id}: No random seed specified. Seeding with datetime.")
        seed = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)  # Fully random
    
    if use_tf:
        return tf.random.Generator.from_seed(seed=seed)
    else:
        return Generator(SFC64(seed=seed))


def load_config(filename: str) -> dict:
    """Try loading config from yaml if os.getcwd() is one level above CartPoleSimulation. This would be the case if CartPoleSimulation is added as Git Submodule in another repository. If not found, load from the current os path.

    :param filename: e.g. 'config.yml'
    :type filename: str
    """
    try:
        config = yaml.load(open(os.path.join("CartPoleSimulation", filename), "r"))
    except FileNotFoundError:
        config = yaml.load(open(filename))
    return config


def load_or_reload_config_if_modified(filepath:str, every:int=5, target_obj=None)->Tuple[Munch,Optional[dict]]:
    """
    Reloads a YAML config if the yaml file was modified since runtime started or since last reloaded.
    The initial call will store the config in a private dict to return on subsequent calls.

    Only modified entries are returned in changes. Added or deleted items are not returned.

    :param filepath the relative path to file. Generate for call with e.g. os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml")
    :param target_obj: an optional object into which the config is assigned for (possibly) use in tensorflow
    :param every: only check every this many times we are invoked

    :returns: (config,changes)
        config: nested Munch of the original config if file has not been modified
            since startup or last reload time,
            otherwise the reloaded config Munch
            The entries are accessed by name, e.g. "CartPole.default.ddweight".
            See https://github.com/Infinidat/munch.
        changes: a dict of key,value of modified config entries in format dict([key,new_value], ...),
            where key is the final element of path a.b..c.key that is in the nested yaml config file
    """
    counter=0
    if filepath in load_or_reload_config_if_modified.counter:
        load_or_reload_config_if_modified.counter[filepath]+=1
        counter=load_or_reload_config_if_modified.counter[filepath]
    else:
        load_or_reload_config_if_modified.counter[filepath] =1
    if filepath in load_or_reload_config_if_modified.cached_configs and counter>0 and counter%every!=0:
        return (load_or_reload_config_if_modified.cached_configs[filepath],None) # if not checking this time, return cached config
    try:
        fp=Path(filepath)
        mtime=fp.stat().st_mtime # get mod time

        if ((not (filepath in load_or_reload_config_if_modified.cached_configs))) \
                or ((filepath in load_or_reload_config_if_modified.cached_configs) and mtime > load_or_reload_config_if_modified.mtimes[filepath]):
            # if loading first time, or we have loaded and the file has been modified since we loaded it, then reload it and flag that it was modified (globally)
            changes = None
            new_config = yaml.load(open(filepath),Loader=yaml.Loader) # loads a nested dict of config file, set loader explicitly to suppress warning about unsafe loader
            new_config_obj=DefaultMunch.fromDict(new_config)
            # now check if there are any changes compared with cached config
            if filepath in load_or_reload_config_if_modified.cached_configs:
                old_config=load_or_reload_config_if_modified.cached_configs[filepath]
                diff=dictdiffer.diff(new_config,old_config) # https://github.com/inveniosoftware/dictdiffer https://stackoverflow.com/questions/32815640/how-to-get-the-difference-between-two-dictionaries-in-python
                for (type,path,change) in diff:
                    if type=='change':
                        (new,old)=change
                        log.info(f'{path} changed: old/new {old}/{new}')
                        if changes is None:
                            changes=dict()
                        key=path.split('.')[-1]
                        changes[key]=new

            load_or_reload_config_if_modified.mtimes[filepath]=mtime
            load_or_reload_config_if_modified.cached_configs[filepath]=new_config_obj
            log.debug(f'(re)loaded modified config (File "{filepath}")') # format (File "XXX") generates pycharm link to file in console output
            if not changes is None and not target_obj is None:
                update_attributes(changes, target_obj)
            return (new_config_obj,changes) # it was modified, so return changes dict
        else:
            return (load_or_reload_config_if_modified.cached_configs[filepath], None) # return previous config and None for changes
    except Exception as e:
        logging.error(f'could not reload {filepath}: got exception {e}')
        raise e

load_or_reload_config_if_modified.cached_configs=dict()
load_or_reload_config_if_modified.mtimes=dict()
load_or_reload_config_if_modified.start_time=time.time()
load_or_reload_config_if_modified.counter=dict()

import numbers
import re

def extract_int(new_value)-> (str,int):
    """  Extracts (possibly signed) int from trailing part of string, e.g. 'ccw-1' -> -1

    :param new_value: str input
    :returns: s,n: leading string and trailing int value or None. If new_value contains /, it is assumed a path and is returned as new_value,None
    """
    str_pat='[a-zA-Z]+'
    int_pat='[-+]?[0-9]+$'
    sp_pat='\/' # file path string
    if re.search(sp_pat,new_value):
        log.debug(f'string {new_value} is a path, will not extract int from it')
        return new_value,None
    string_name=re.search(str_pat,new_value).group(0)
    int_match=re.search(int_pat, new_value)
    int_val=int(int_match.group(0)) if int_match else None
    return string_name,int_val


def update_attributes(updated_attributes: "dict[str, TensorType]", target_obj):
    """ Update attributes in compiled code (tensorflow JIT) that have changed, i.e. copy them to the compiled/GPU instance.

    After this call, such attribute values are available to the TF function as self.key, where key is the key used in dict.

    Note that this attribute will NOT be accessible the compiled TF code if it has been declared ahead of time. I.e., do not define e.g. self.a=something and then
    expect config a: to be picked up by the compiled code. It must be set for the first time by update_attributes!

    Used in various controllers in Control_Toolkit/Control_Toolkit_ASF_Template/Controllers.

    :param updated_attributes: a dict [key,value] with string key 'attribute' (aka property) and scalar numeric value or string value to set.
        If the value is a string and it ends with an intteger, e.g. 'spin0', then the
        attribute is set to 'spin' and attrribute_number is set to the int value of the number.
    :param target_obj: the object into which we set the attribute.

    """
    for property, new_value in updated_attributes.items():
        if hasattr(target_obj, property) :  # make sure it is mutable if we want to set it
            objtype=None
            if isinstance(new_value,numbers.Integral):
                objtype=target_obj.lib.int32
            elif isinstance(new_value,numbers.Real):
                objtype=target_obj.lib.float32
            elif isinstance(new_value,str):
                objtype=target_obj.lib.string
            elif isinstance(new_value,np.ndarray):
                objtype=target_obj.lib.float32  # todo assuming all np arrays should go to float 32 here
            else:
                log.warning(f'attribute "{property}" has unknown object type {type(new_value)}; cannot assign it')
            if not objtype is None and objtype!=target_obj.lib.string:
                try:
                    target_obj.lib.assign(getattr(target_obj, property), target_obj.lib.to_variable(new_value,objtype))
                except ValueError:
                    log.error(f'target attribute "{property}" is probably float type but in config file it is int. Add a trailing "." to the number "{new_value}"')
                    # target_obj.lib.assign(getattr(target_obj, property), target_obj.lib.to_variable(new_value, target_obj.lib.to_variable(float(new_value), target_obj.lib.float32)))
            else: # string type; if it ends with int, then also assign a name_number variable
                val_str,val_int = extract_int(new_value)
                if isinstance(val_int,int):
                    property_number_name = property + '_number'
                    target_obj.lib.assign(getattr(target_obj, property), target_obj.lib.to_variable(val_str, objtype))
                    target_obj.lib.assign(getattr(target_obj, property_number_name),
                                          target_obj.lib.to_variable(val_int, target_obj.lib.int32))
                else: # just assign it if it does not end with digit
                    target_obj.lib.assign(getattr(target_obj, property),
                                          target_obj.lib.to_variable(new_value, objtype))

        else:
            log.debug(
                f"tensorflow attribute '{property}' does not exist in {target_obj.__class__.__name__}, setting it for first time")
            if target_obj.lib is None:
                setattr(target_obj, property, new_value)
            else:
                # just set the attribute, don't assign in (like =) since some immutable objects cannot be assigned
                if isinstance(new_value,numbers.Integral):
                    setattr(target_obj, property, target_obj.lib.to_variable(new_value, target_obj.lib.int32))
                elif isinstance(new_value, str):
                    # create a string tensor with base name, and another tf.variable named name_number, e.g. policy='dance' and policy_number=0
                    val_str,val_int = extract_int(new_value)
                    if isinstance(val_int,int):
                        property_number_name = property + '_number'
                        setattr(target_obj, property,target_obj.lib.to_variable(val_str, target_obj.lib.string)) # set string var
                        setattr(target_obj, property_number_name,target_obj.lib.to_variable(val_int, target_obj.lib.int32)) # set int code var
                        log.debug(f"setting tensorflow attribute '{property}={val_str}' and {property_number_name}={int(val_int)}")
                    else:  # just assign it if it does not end with digit
                        setattr(target_obj, property, target_obj.lib.to_variable(new_value, target_obj.lib.string))
                elif isinstance(new_value,numbers.Real) or isinstance(new_value,np.ndarray):
                    setattr(target_obj, property, target_obj.lib.to_variable(new_value, target_obj.lib.float32))
                else:
                    log.warning(f'type "{type(new_value)}" of attribute "{property}" is not settable type, must be int, float (or np.ndarray), or string')


class MockSpace:
    def __init__(self, low, high, shape: Tuple, dtype=np.float32) -> None:
        self.low, self.high = np.atleast_1d(low).astype(dtype), np.atleast_1d(high).astype(dtype)
        self.dtype = dtype
        self.shape = shape
        

timers = {}
times = {}
class Timer:
    def __init__(self, timer_name='', delay=None, show_hist=False, numpy_file=None):
        """ Make a Timer() in a _with_ statement for a block of code.
        The timer is started when the block is entered and stopped when exited.
        The Timer _must_ be used in a with statement.
        :param timer_name: the str by which this timer is repeatedly called and which it is named when summary is printed on exit
        :param delay: set this to a value to simply accumulate this externally determined interval
        :param show_hist: whether to plot a histogram with pyplot
        :param numpy_file: optional numpy file path
        """
        self.timer_name = timer_name
        self.show_hist = show_hist
        self.numpy_file = numpy_file
        self.delay=delay

        if self.timer_name not in timers.keys():
            timers[self.timer_name] = self
        if self.timer_name not in times.keys():
            times[self.timer_name]=[]

    def __enter__(self):
        if self.delay is None:
            self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.delay is None:
            self.end = time.time()
            self.interval = self.end - self.start  # measured in seconds
        else:
            self.interval=self.delay
        times[self.timer_name].append(self.interval)

    def print_timing_info(self, logger=None):
        """ Prints the timing information accumulated for this Timer
        :param logger: write to the supplied logger, otherwise use the built-in logger
        """
        if len(times)==0:
            log.error(f'Timer {self.timer_name} has no statistics; was it used without a "with" statement?')
            return
        a = np.array(times[self.timer_name])
        timing_mean = np.mean(a) # todo use built in print method for timer
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        s='{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(self.timer_name, len(a),
                                                                      eng(timing_mean), eng(timing_std),
                                                                      eng(timing_median), eng(timing_min),
                                                                      eng(timing_max))

        if logger is not None:
            logger.info(s)
        else:
            log.info(s)

def print_timing_info():
    for k,v in times.items():  # k is the name, v is the list of times
        a = np.array(v)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        log.info('== Timing statistics from all Timer ==\n{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(k, len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))
        if timers[k].numpy_file is not None:
            try:
                log.info(f'saving timing data for {k} in numpy file {timers[k].numpy_file}')
                log.info('there are {} times'.format(len(a)))
                np.save(timers[k].numpy_file, a)
            except Exception as e:
                log.error(f'could not save numpy file {timers[k].numpy_file}; caught {e}')

        if timers[k].show_hist:

            def plot_loghist(x, bins):
                hist, bins = np.histogram(x, bins=bins) # histogram x linearly
                if len(bins)<2 or bins[0]<=0:
                    log.error(f'cannot plot histogram since bins={bins}')
                    return
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins)) # use resulting bin ends to get log bins
                plt.hist(x, bins=logbins) # now again histogram x, but with the log-spaced bins, and plot this histogram
                plt.xscale('log')

            dt = np.clip(a,1e-6, None)
            # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            try:
                plot_loghist(dt,bins=100)
                plt.xlabel('interval[ms]')
                plt.ylabel('frequency')
                plt.title(k)
                plt.show()
            except Exception as e:
                log.error(f'could not plot histogram: got {e}')


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info) 
