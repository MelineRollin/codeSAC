import gc
import torch


def gc_all():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    return 0


def gc_n():
    n = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                n += 1
        except:
            pass
    return n


def gc_elem():
    n = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                n += torch.numel(obj)
        except:
            pass
    return n


def gc_simple_n():
    return len(gc.get_objects())
