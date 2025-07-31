import logging
import uuid
import time
import json

from .dictobj import ConfigurationBase


def callFunctionBeginLogger(nameFunction: str, fun=None, arg: str = None, option: tuple = None, includeOption: bool = True):
    if fun is None:
        fun = logging.info
    if arg is None:
        arg = ""
    st = time.time()
    id = f"{nameFunction}_{uuid.uuid4().hex}"
    msg = f"[Call function {id}] {arg}"
    fun(msg)
    if option is not None and includeOption:
        try:
            if isinstance(option, ConfigurationBase):
                option = option.to_dict()
            msg = f"[Option function {id}] {json.dumps(option)}"
            fun(msg)
        except:
            pass
    return {"id": id, "st": st}


def traceFunctionLogger(data: dict, msg: str, fun=None):
    if fun is None:
        fun = logging.info

    id = data["id"]
    msg = f"[Trace function {id}] {msg}"
    fun(msg)


def callFunctionEndLogger(data: dict, fun=None):
    if fun is None:
        fun = logging.info

    et = time.time()
    id = data["id"]
    dt = et - data["st"]
    msg = f"[Call function {id}] processed in {str(dt)}s"
    fun(msg)


def callFunctionEndErrorLogger(data: dict, err: str, fun=None):
    if fun is None:
        fun = logging.info

    et = time.time()
    id = data["id"]
    dt = et - data["st"]
    msg = f"[Call function {id}] processed in {str(dt)}s but an error occured: {err}"
    fun(msg)
