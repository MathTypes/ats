import datetime

def round_down(tm, minutes):
    discard = datetime.timedelta(minutes=tm.minute % minutes,
                                seconds=tm.second,
                                microseconds=tm.microsecond)
    tm -= discard
    return tm

def round_up(tm, minutes):
    discard = datetime.timedelta(minutes=tm.minute % minutes,
                                seconds=tm.second,
                                microseconds=tm.microsecond)
    new_tm = tm - discard
    if new_tm != tm:
        new_tm += datetime.timedelta(minutes=minutes)

    return new_tm

    