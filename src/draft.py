from functools import cmp_to_key


imgs = ["za", "abcd", "zz", "zc", "bb05", "bb01"]

def compare_path(item1: str, item2: str):
    return_value = 1

    if len(item1) > len(item2):
        return_value = 1
    elif len(item1) < len(item2):
        return_value = -1
    else:
        if item1 > item2:
            return_value = 1
        elif item1 < item2:
            return_value = -1
        else:
            return_value = 0

    return return_value

imgs.sort(key=cmp_to_key(compare_path))
print(imgs)