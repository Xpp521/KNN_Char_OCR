from os import startfile
from sys import platform
from subprocess import call


def open_file(filepath):
    """
    Open file with system default program.
    :param filepath: the file path.
    """
    if platform.startswith(('win', 'cygwin')):  # Windows
        startfile(filepath)
    elif platform.startswith('linux'):          # Linux
        call(['xdg-open', filepath])
    elif platform.startswith('darwin'):         # Mac OS
        call(['open', filepath])
    else:
        print('【{}】open fail! unknown system platform.'.format(filepath))
