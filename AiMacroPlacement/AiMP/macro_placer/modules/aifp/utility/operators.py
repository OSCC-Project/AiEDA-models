def number_digits(number:float)->int:
    """get digits of number"""
    temp = abs(int(number))
    count = 0
    while temp != 0:
        count += 1
        temp //= 10
    return count

def scale(number:float, digits:int=3)->float:
    """scaling positive number to 10^(digits-1) ~ 10^(digits)"""
    if number == 0:
        return number
    if number < 0:
        print('error number: ', number)
        raise ValueError('input number should be positive')
    origin_digits = number_digits(number)
    return number * 10**(digits-origin_digits)

def flip_horizontal(orient:str):
    if orient == 'N':
        return 'FN'
    elif orient == 'FN':
        return 'N'
    elif orient == 'S':
        return 'FS'
    elif orient == 'FS':
        return 'S'
    elif orient == 'W':
        return 'FW'
    elif orient == 'FW':
        return 'W'
    elif orient == 'E':
        return 'FE'
    elif orient == 'FE':
        return 'E'
    else:
        raise ValueError('input orient error!, orient: {}'.format(orient))

def flip_vertical(orient):
    if orient == 'N':
        return 'FS'
    elif orient == 'FS':
        return 'N'
    elif orient == 'FN':
        return 'S'
    elif orient == 'S':
        return 'FN'
    elif orient == 'W':
        return 'FE'
    elif orient == 'FE':
        return 'W'
    elif orient == 'FW':
        return 'E'
    elif orient == 'E':
        return 'FW'
    else:
        raise ValueError('input orient error!, orient: {}'.format(orient))

def get_mirror_operation(origin_orient:str, new_orient:str):
    # assert only allows mirror operation...
    if origin_orient == new_orient:
        return None

    if new_orient == flip_horizontal(origin_orient):
        operation = 'MY'
    elif new_orient == flip_vertical(origin_orient):
        operation = 'MX'
    elif new_orient == flip_horizontal(flip_vertical(origin_orient)):
        operation = 'MXMY'
    else:
        raise RuntimeError('orient error, only mirror operation allowed...')
    return operation

def get_offset(
        origin_orient:str,
        new_orient:str,
        width:float,
        height:float,
        origin_offset_x:float,
        origin_offset_y:float):
    """return:
        new_offset_x : float
        new_offset_y : float"""

    if origin_orient == new_orient:
        return origin_offset_x, origin_offset_y

    operation = get_mirror_operation(origin_orient, new_orient)

    if operation == 'MX':
        new_offset_x = origin_offset_x
        new_offset_y = height - origin_offset_y
    
    elif operation == 'MY':
        new_offset_x = width - origin_offset_x
        new_offset_y = origin_offset_y
    
    elif operation == 'MXMY':
        new_offset_x = width - origin_offset_x
        new_offset_y = height - origin_offset_y

    else:
        raise RuntimeError('operation unknown')
    
    return new_offset_x, new_offset_y



# if __name__ == '__main__':
#     # wl = 25211396.0
#     # print(number_digits(wl))
#     # print(scale(wl))
#     # rewards = - scale(wl)
#     # print('rewards:', rewards)
#     # print(-10**(number_digits(rewards)))

#     print(get_mirror_operation('FW', 'W'))
#     print(get_mirror_operation('W', 'FW'))
#     print(get_mirror_operation('E', 'FE'))
#     print(get_mirror_operation('N', 'FN'))
#     print(get_mirror_operation('N', 'FS'))
#     print(get_mirror_operation('FN', 'FS'))
#     print(get_mirror_operation('S', 'N'))