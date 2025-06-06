def rating_count(price_tag):
    if price_tag >= 1000:
        if ((price_tag < 2500) and (price_tag >= 1900)):
            return 4.95
        else:
            return 5
    if ((price_tag < 1000) and (price_tag >= 100)):
        if (price_tag > 630):
            return 5
        if ((price_tag < 630) and (price_tag >= 390)):
            return 3
        if ((price_tag < 390) and (price_tag >= 300)):
            return 3.5
        if ((price_tag < 300) and (price_tag >= 250)):
            return 4
        if ((price_tag < 250) and (price_tag >= 200)):
            return 4.5
        if ((price_tag < 200) and (price_tag >= 100)):
            return 3.33
    else:
        if (price_tag >= 85):
            return 4.7
        if (price_tag < 85 and price_tag >= 50):
            return 3.9
        if (price_tag < 50 and price_tag >= 20):
            return 2
        if (price_tag < 20):
            return 4.7


def fees_extraction(s):
    if '-' in s:
        data = ''.join(' '.join(s.split("-")).split('RM')).strip().split(' ')
        data = 0.5*(float(data[0]) + float(data[3]))
        return data
    elif (s == 'unavailable' or s == 'Unsupported Address'):
        return 0
    else:
        data = float(s.strip('RM'))
        return data
