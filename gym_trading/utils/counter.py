'''
- 8 tage, alle in untereinander in df
- pro tag sollen insg 1_000_000 trainingsteps gemacht werden
- wenn ende des tages erreicht wird soll local_step_number wieder mit index der zu dem aktuellen Tag gehoert springen

(1) loop over each day until timesteps_per_day is reached
(2) loop over entire env until timestep (?) is reached

TODO: timesteps_per_day is not precise, maybe number of episodes per day better?
'''

class Counter:
    def __init__(self, day_indices_dict: int, timesteps_per_day: int or None,
                 loop_over_day: bool = False):
        self.day_indices_dict = day_indices_dict
        self.timesteps_per_day = timesteps_per_day # how many steps the agent should train on one day
        self.loop_over_day = loop_over_day
        self.global_step_counter = 0 # how many steps have been taken in total
        if self.loop_over_day:
            self.day_step_counter = 0 # how many steps have been taken in this day
        self.local_step_number = 0  # current index in df, corresponds to local_step_number
        self.day_number = 0 # on which day the agent currently trains, key for dict

    def set_local_step_number(self, local_step_number):
        assert self.local_step_number + 1 == local_step_number, \
            f'self.local_step_number+1 is {self.local_step_number+1} and local_step_number is {local_step_number}'
        self.local_step_number = local_step_number
        
        if self.loop_over_day:
            self.day_step_counter += 1
        self.global_step_counter += 1

    def set_local_step_number_in_reset(self, local_step_number):
        self.local_step_number = local_step_number

    def env_reset_call(self):
        if self.loop_over_day and self.day_step_counter < self.timesteps_per_day:
            # most simple way, next episode is still same day
            low, high = self._get_current_day_low_high_index()
            rand_high = (high-low)//5
            rand_high += low
            return low, rand_high, high
        elif (self.loop_over_day and self.day_step_counter >= self.timesteps_per_day) \
            or not self.loop_over_day:
            # jump to next day
            self.day_number = (self.day_number+1)%len(self.day_indices_dict) # prevents index error, just jumps to first day
            if self.loop_over_day:
                self.day_step_counter = 0
            low, high = self._get_current_day_low_high_index()
            rand_high = (high-low)//5
            rand_high += low
            return low, rand_high, high
        else:
            raise IndexError('Something went wrong with the indices, check!')

    def get_max_steps(self):
        # returns index of first day
        return self.day_indices_dict[self.day_number][1]
    
    def _get_current_day_low_high_index(self):
        low = self.day_indices_dict[self.day_number][0]
        high = self.day_indices_dict[self.day_number][1] # corresponds to max_steps
        return low, high
    



    