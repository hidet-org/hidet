# %%
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def quote_green(s):
    return bcolors.OKGREEN + s + bcolors.ENDC


def quote_red(s):
    return bcolors.FAIL + s + bcolors.ENDC


def quote_cyan(s):
    return bcolors.OKCYAN + s + bcolors.ENDC


def quote_warning(s):
    return bcolors.WARNING + s + bcolors.ENDC


def quote_fail(s):
    return bcolors.FAIL + s + bcolors.ENDC
