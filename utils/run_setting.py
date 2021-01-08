import os
import time


def get_run_id() -> str:
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
    return time.strftime("%Y%m%d_%H%M%S")

