from abc import ABC
import time

class Watch(ABC):
    tstart = None

    @classmethod
    def start(cls):
        cls.tstart = time.time()


    @classmethod
    def stop(cls):
        tstop = (time.time() - cls.tstart)*1000
        cls.tstart = None
        return tstop


    @classmethod
    def stop_print(cls):
        print(f"Ellapsed time {cls.stop()} ms.")



