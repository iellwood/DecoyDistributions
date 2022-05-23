"""
Code for printing the current progress to the console, "Validating density estimation models with decoy distributions"

Written in tensorflow 1.15
"""

import numpy as np
from time import time


class ProgressTracker:


    def __init__(self, total_steps):
        """
        This helper class prints steps to completion and the amount of time taken so far.

        Run:

        ``progress_tracker = Progress_tracker(total_steps)``

        just before training and

        ``progress_tracker.progress(step)``

        when you want to print to the console

        :param total_steps: Total number of steps being peformed
        """
        self.total_steps = total_steps
        self.t_0 = time()
        self.t_last_step = self.t_0

    def progress(self, step):
        '''
        Prints step/total_steps and the amount of time taken so far.

        :param step: Current step
        '''
        t = time()
        total_elapsed_time = t - self.t_0
        return 'Completed ' + str(step) + '/' + str(self.total_steps) + ' in ' + self._time_formatted(total_elapsed_time)


    @staticmethod
    def _time_formatted(t):
        """
        Turns a time in seconds into a str listing days, minutes, hours and seconds.

        :param t: time, in seconds
        :return: str listing the time in days, hours, minutes and seconds.
        """
        days = int(np.floor(t/(24*3600)))
        hours = int(np.floor((t/3600) % 24))
        mins = int(np.floor((t/60) % 60))
        secs = int(np.floor(t % 60))

        list_of_strings = []
        non_zero_found = False
        if days > 0:
            list_of_strings.append(str(days) + 'd, ')
            non_zero_found = True
        if hours > 0 or non_zero_found:
            list_of_strings.append(str(hours) + 'h, ')
            non_zero_found = True

        if mins > 0 or non_zero_found:
            list_of_strings.append(str(mins) + 'm, ')

        list_of_strings.append(str(secs) + 's')

        return ''.join(list_of_strings)








