import time


def print_current_time():
    """
    打印当前时间，精确到毫秒
    """
    t = time.time()
    sec = time.localtime(t)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', sec)
    ms = int((t - int(t)) * 1000)
    print(f'{time_str}.{ms:03d}')