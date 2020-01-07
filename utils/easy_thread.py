#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple way to create basic decorrelated threads with Python 3."""

from threading import Thread
from enum import Enum, IntEnum

DAEMON = False

class TaskState(Enum):
    """Represents task states."""

    NOT_STARTED = 0
    RUNNING = 1
    FAILED = -1

class DebugLevel(IntEnum):
    """Represents debug levels."""

    ERROR = 0
    WARNING = 1
    INFO = 2

DEBUG_LEVEL = DebugLevel.INFO

def LOSE_RESULT(value):
    """Do nothing with the result."""
    pass

class EasyThread(Thread):
    """Easy thread instance.
    Run the specified function with specified args, and output result by
    calling the specified set_result function."""

    def __init__(
        self, name, func, args, kwargs, debug_level=DEBUG_LEVEL,
        set_result=LOSE_RESULT, write_log=print):
        """Init method."""
        super().__init__(name=name)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._debug_level = debug_level
        self._set_result = set_result
        self._write_log = write_log

    if True:  # to fold all the properties

        def get_func(self):
            """func getter."""
            return self._func
        
        def set_func(self, new_value):
            """func setter."""
            raise ValueError("func cannot be changed after initialization.")
        
        func = property(get_func, set_func)

        def get_args(self):
            """args getter."""
            return self._args

        def set_args(self, new_value):
            """args setter."""
            raise ValueError("args cannot be changed after initialization.")

        args = property(get_args, set_args)

        def get_kwargs(self):
            """kwargs getter."""
            return self._kwargs

        def set_kwargs(self, new_value):
            """kwargs setter."""
            raise ValueError("kwargs cannot be changed after initialization.")

        kwargs = property(get_kwargs, set_kwargs)

        def get_debug_level(self):
            """debug_level getter."""
            return self._debug_level

        def set_debug_level(self, new_value):
            """debug_level setter."""
            raise ValueError("debug_level cannot be changed after initialization.")

        debug_level = property(get_debug_level, set_debug_level)

        def get_set_result(self):
            """set_result getter."""
            return self._set_result

        def set_set_result(self, new_value):
            """set_result setter."""
            raise ValueError("set_result cannot be changed after initialization.")

        set_result = property(get_set_result, set_set_result)

        def get_write_log(self):
            """write_log getter."""
            return self._write_log

        def set_write_log(self, new_value):
            """write_log setter."""
            raise ValueError("write_log cannot be changed after initialization.")

        write_log = property(get_write_log, set_write_log)

    def run(self):
        """Main method of the thread."""
        self.log(DebugLevel.INFO, "--> Starting")
        self._set_result(TaskState.RUNNING)
        try:
            result = self._func(*self.args, **self.kwargs)
            self._set_result(result)
        except Exception as e:
            self.log(
                DebugLevel.ERROR, "<!> Error during runtime: <{}>".format(e))
            self._set_result(TaskState.FAILED)
        self.log(DebugLevel.INFO, "<-- Exiting")

    def log(self, level, msg):
        """Prints msg iif level <= self._debug_level."""
        if not isinstance(level, DebugLevel):
            raise TypeError("Log level should be a DebugLevel enum value.")
        if level <= self._debug_level:
            self._write_log("[Thread {}] {}".format(self.name, msg))

class ThreadPool(object):
    """Represents a thread pool, which can accept arbitrary number of threads
    running at the same time."""

    def __init__(
        self, daemon=DAEMON, debug_level=DEBUG_LEVEL, write_log=print):
        """Init method."""
        super().__init__()
        self._daemon = daemon
        self._debug_level = debug_level
        self._write_log = write_log
        self._registry = {}

    if True:  # to fold all the properties

        def get_daemon(self):
            """daemon getter."""
            return self._daemon

        def set_daemon(self, new_value):
            """daemon setter."""
            raise ValueError("daemon cannot be changed after initialization.")

        daemon = property(get_daemon, set_daemon)

        def get_debug_level(self):
            """debug_level getter."""
            return self._debug_level

        def set_debug_level(self, new_value):
            """debug_level setter."""
            raise ValueError("debug_level cannot be changed after initialization.")

        debug_level = property(get_debug_level, set_debug_level)

        def get_write_log(self):
            """write_log getter."""
            return self._write_log

        def set_write_log(self, new_value):
            """write_log setter."""
            raise ValueError("write_log cannot be changed after initialization.")

        write_log = property(get_write_log, set_write_log)

        def get_registry(self):
            """registry getter."""
            return self._registry

        def set_registry(self, new_value):
            """registry setter."""
            raise ValueError(
                "registry cannot be changed after initialization.")

        registry = property(get_registry, set_registry)

    def add(self, key, func, *args, **kwargs):
        """Creates and executes a new thread with the specified data."""
        if key in self._registry:
            raise ValueError("Key <{}> already in registry.".format(key))
        self._registry[key] = TaskState.NOT_STARTED
        new_thread = EasyThread(
            str(key), func, args, kwargs,
            debug_level=self._debug_level,
            set_result=self.build_set_result(key),
            write_log=self._write_log)
        new_thread.daemon = self._daemon
        new_thread.start()
        # So that add calls can be chained. FP power my friends :)
        return self

    def build_set_result(self, key):
        """Builds the set_result function for the specified thread."""

        def set_result(value):
            """Send the result of the computation to the parent."""
            self._registry[key] = value

        return set_result

    def has_started(self, key):
        """Says if the specified thread has started."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        return self._registry[key] != TaskState.NOT_STARTED

    def is_running(self, key):
        """Says if the specified thread is running."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        return self._registry[key] == TaskState.RUNNING

    def has_finished(self, key):
        """Says if the specified thread has finished."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        result = self._registry[key]
        return (
            result != TaskState.NOT_STARTED and
            result != TaskState.RUNNING)

    def has_succeeded(self, key):
        """Says if the specified thread has succeeded."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        result = self._registry[key]
        return (
            result != TaskState.NOT_STARTED and
            result != TaskState.RUNNING and
            result != TaskState.FAILED)

    def has_failed(self, key):
        """Says if the specified thread has failed."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        return self._registry[key] == TaskState.FAILED

    def get_result(self, key):
        """Returns the result of the specified thread."""
        if key not in self._registry:
            raise IndexError("Specified key not present in the registry.")
        return self._registry[key]

def main():
    """Examples."""
    from time import sleep

    # Example 1
    from time import sleep
    from random import random
    def print_n(author, n):
        """Prints number from 0 to n - 1."""
        for i in range(n):
            sleep(random())
            print("[{}]: {}".format(author, i))
    ThreadPool().add(1, print_n, 1, 10).add(2, print_n, 2, 8)

    # So that examples won't collide
    sleep(10)
    # sleep well baby

    # Example 2
    from urllib.request import urlopen
    from time import sleep
    def get_page():
        """Returns google homepage."""
        return urlopen("https://www.google.com").read()
    pool = ThreadPool()
    print("Asking for the page...")
    pool.add(0, get_page)
    count = 0
    while pool.is_running(0):
        count += 1
        print("    ...waiting for the page [{}]".format(count))
        sleep(0.01)
    if pool.has_failed(0):
        print("Unable to get the page :(")
    else:
        page = pool.get_result(0)
        print("Page received: {}".format(page[:min(20, len(page))]))

if __name__ == "__main__":
    main()
