import sys

# 'sys' provides access to system-specific parameters and functions that interact with the Python runtime environment.
# ih this project, iam using it to access the error details of the runtime


def error_message_detail(error, sys_obj: sys):
    _, _, exc_tb = sys_obj.exc_info()
    # exc_tb contains all the info about the exceptions
    file_name = exc_tb.tb_frame.f_code.co_filename
    # gives the filename where exception is raised
    line_num = exc_tb.tb_lineno
    # gives the line number in the file where exception is raised
    detailed_err_msg = "Error occured in python script[{0}] line number [{1}] error message [{2}]".format(
        file_name, line_num, str(error)
    )
    return detailed_err_msg


class CustomException(Exception):
    def __init__(self, error_message, sys_obj: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, sys_obj=sys_obj)

    # __str__ = dunder method: it returns the string representation of the object of CustomException whenever print(exception_obj) or raised
    def __str__(self):
        return self.error_message
