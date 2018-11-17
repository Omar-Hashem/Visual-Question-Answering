import threading

# Sub-Class for calling a function in a new thread
class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args
        self._ret_val = None
        self.start()

    def run(self):
        self._ret_val = self._target(*self._args)

    def get_ret_val(self):
        self.join()
        return self._ret_val
