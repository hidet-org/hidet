import matplotlib.font_manager
flist = matplotlib.font_manager.get_fontconfig_fonts()

names = []

for fname in flist:
    try:
        font_name = matplotlib.font_manager.FontProperties(fname=fname).get_name()
    except:
        continue
    names.append(font_name)

with open('./font_names.txt', 'w') as f:
    f.write('\n'.join(names))

for i in range(3):
    pass

