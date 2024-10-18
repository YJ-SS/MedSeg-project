def log_print(status: str, content:str)-> None:
    assert  status == 'INFO' or status == 'WARNING' or status == 'CRITICAL' or status == 'ERROR',\
    log_print("ERROR", "Status must in ['INFO','WARNING','CRITICAL','ERROR'], "
                       "current status: {0}".format(status))
    if status == 'INFO':
        print("\033[0;32;1mINFO:\033[0m", content)
    elif status == 'WARNING':
        print("\033[0;33;1mWARNING:\033[0m", content)
    elif status == 'ERROR':
        print("\033[0;31;1mERROR:\033[0m", content)
    elif status == 'CRITICAL':
        print("\033[0;35;1mCRITICAL:\033[0m", content)

