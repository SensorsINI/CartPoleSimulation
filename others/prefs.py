# from __future__ import print_function
from typing import Any

from pypref import SinglePreferences as PREF
class MyPreferences(PREF):
    """ Class to hold user preference values, adds put() method to pypref SinglePreferences"""
    # *args and **kwargs can be replaced by fixed arguments
    def put(self,key:str, value:Any):
        """ Put a value to preferences using key
        :param key: a string key
        :param value: the value to put
        """
        self.update_preferences({key:value})