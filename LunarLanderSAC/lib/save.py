import datetime
import os
import torch

prefix = "run"


def time_stamp(dt):
    y = str(dt.year)[-2:]
    mon = str(dt.month).rjust(2, '0')
    d = str(dt.day).rjust(2, '0')
    h = str(dt.hour).rjust(2, '0')
    minute = str(dt.minute).rjust(2, '0')
    sec = str(dt.second).rjust(2, '0')

    return f"{y}{mon}{d}-{h}{minute}{sec}"


def create_run_folder(run_name):
    dt = datetime.datetime.now()
    dt_str = time_stamp(dt)

    folder_name = f"{prefix}/{dt_str} - {run_name}"
    os.mkdir(f"{folder_name}")

    return folder_name


def save_policy(model, folder):
    dt = datetime.datetime.now()
    dt_str = time_stamp(dt)
    file_name = f"policy network - {dt_str}.ptm"
    torch.save(model.state_dict(), f"{folder}/f{file_name}")
    print(f"policy network has been saved to \"{file_name}\"")
