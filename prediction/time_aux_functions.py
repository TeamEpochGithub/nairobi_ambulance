
from datetime import datetime, timedelta, time
import random

# I don't think this is the most efficient way to do it, but it will solve in for now. Rethink it
def roundDateTime3h(datetime):
    if isNowInTimePeriod(time(22, 30), time(1, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=0)

    elif isNowInTimePeriod(time(1, 30), time(4, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=3)
    elif isNowInTimePeriod(time(4, 30), time(7, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=6)
    elif isNowInTimePeriod(time(7, 30), time(10, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=9)
    elif isNowInTimePeriod(time(10, 30), time(13, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=12)
    elif isNowInTimePeriod(time(13, 30), time(16, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=15)
    elif isNowInTimePeriod(time(16, 30), time(19, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=18)
    elif isNowInTimePeriod(time(19, 30), time(22, 30), time(datetime.hour, datetime.minute)):
        datetime = datetime.replace(hour=21)

    datetime = datetime.replace(minute=0)
    datetime = datetime.replace(second=0)
    datetime = datetime.replace(microsecond=0)
    return datetime


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:  # Over midnight
        return nowTime >= startTime or nowTime <= endTime


def gen_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()
