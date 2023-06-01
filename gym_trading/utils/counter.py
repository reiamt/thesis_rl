'''
- 8 tage, alle in untereinander in df
- pro tag sollen insg 1_000_000 trainingsteps gemacht werden
- wenn ende des tages erreicht wird soll local_step_number wieder mit index der zu dem aktuellen Tag gehoert springen
TODO: prüfen, dass day_number nie grosser wird als anzahl der tage die wir haben
'''

class Counter:
    def __init__(self, day_indices_dict: int, timesteps_per_day: int):
        self.day_indices_dict = day_indices_dict
        self.timesteps_per_day = timesteps_per_day # how many steps the agent should train on one day
        self.global_step_counter = 0 # how many steps have been taken in total
        self.day_step_counter = 0 # how many steps have been taken in this day
        self.local_step_number = 0  # current index in df, corresponds to local_step_number
        self.day_number = 0 # on which day the agent currently trains, key for dict

    def set_local_step_number(self, local_step_number):
        assert self.local_step_number + 1 == local_step_number, \
            f'self.local_step_number+1 is {self.local_step_number+1} and local_step_number is {local_step_number}'
        self.local_step_number = local_step_number

        self.day_step_counter += 1
        self.global_step_counter += 1

    def set_local_step_number_in_reset(self, local_step_number):
        self.local_step_number = local_step_number

    def env_reset_call(self):
        if self.day_step_counter < self.timesteps_per_day:
            # most simple way
            low, high = self._get_current_day_low_high_index()
            rand_high = (high-low)//5
            rand_high += low
            return low, rand_high, high
        elif self.day_step_counter >= self.timesteps_per_day:
            # jump to next day
            self.day_number += 1
            self.day_step_counter = 0
            low, high = self._get_current_day_low_high_index()
            rand_high = (high-low)//5
            rand_high += low
            return low, rand_high, high

    def get_max_steps(self):
        '''
        returns index of last observation in current day (day_number)
        '''
        return self.day_indices_dict[self.day_number][1]
    
    def _get_current_day_low_high_index(self):
        low = self.day_indices_dict[self.day_number][0]
        high = self.day_indices_dict[self.day_number][1] # corresponds to max_steps
        return low, high
    



    