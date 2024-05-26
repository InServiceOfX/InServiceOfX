from ..Schedulers import CreateSchedulerMap

def change_scheduler_or_not(pipe, scheduler_name=None):
    if scheduler_name == None:
        return False
    schedulers_map = CreateSchedulerMap.get_map()
    pipe.scheduler = schedulers_map[scheduler_name].from_config(
        pipe.scheduler.config)

    return True