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


# borrowed from https://stackoverflow.com/a/13565185
# as noted there, the calendar module has a function of its own
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)

def monthlist(begin,end):
    #begin = datetime.datetime.strptime(begin, "%Y-%m-%d")
    #end = datetime.datetime.strptime(end, "%Y-%m-%d")

    result = []
    while True:
        if begin.month == 12:
            next_month = begin.replace(year=begin.year+1,month=1, day=1)
        else:
            next_month = begin.replace(month=begin.month+1, day=1)
        if next_month > end:
            break
        result.append ([begin, last_day_of_month(begin)])
        begin = next_month
    result.append ([begin, end])
    return result

    
