from .temphandler import TempHandler
import os
import random


class WorkspaceInfo:
    __log_path: str
    __output_path: str
    __temp_path: str
    __base_path: str
    __backup_path: str
    __idSession: str

    __temp_handler: TempHandler = None

    __logLevel = 'INFO'

    @property
    def log_path(self):
        return self.__log_path

    @property
    def isVerbose(self):
        return self.__logLevel == "DEBUG"

    @property
    def base_path(self):
        return self.__base_path

    @property
    def output_path(self):
        return self.__output_path

    @property
    def temp_path(self):
        return self.__temp_path

    @property
    def backup_path(self):
        return self.__backup_path

    @property
    def id_session(self):
        return self.__idSession

    def __init__(self, baseFolder: str = "./", idSession: str = None):
        self.change_base_folder(baseFolder)
        if idSession is not None:
            self.__idSession = idSession
        else:
            self.__idSession = str(random.randint(1000, 9999))

    def state_logger(self, useVerbose: bool):
        if useVerbose:
            self.__logLevel = "DEBUG"
        else:
            self.__logLevel = "INFO"

    def get_temp_handler(self):
        if self.__temp_handler is None:
            self.__temp_handler = TempHandler(self.temp_path, reset=True)
        return self.__temp_handler

    def get_folder(self, name: str = ".") -> str:
        p = None
        if name == "log":
            p = self.__log_path
        elif name == "outputs":
            p = self.__output_path
        elif name == "backup":
            p = self.__backup_path
        elif name == "temp":
            p = self.__temp_path
        elif name in [".", "./", "base"]:
            p = self.__base_path

        if not os.path.exists(p):
            os.makedirs(p)
        return p

    def get_complete_path(self,
                        folderName: str,
                        fileName: str,
                        isDic=False) -> str:
        p = self.get_folder(folderName)
        if fileName is None:
            f = p
            isDic = True
        elif isinstance(fileName, list) or isinstance(fileName, tuple):
            f = os.path.join(p, *fileName)
        else:
            f = os.path.join(p, fileName)

        if isDic and not os.path.exists(f):
            os.makedirs(f, exist_ok=True)
        elif not isDic:
            basef = os.path.dirname(f)
            if not os.path.exists(basef):
                os.makedirs(basef)
        return f

    def change_base_folder(self, folder: str):
        self.__base_path = folder
        self.__log_path = os.path.join(folder, "logs")
        self.__output_path = os.path.join(folder, "outputs")
        self.__temp_path = os.path.join(folder, "temp")
        self.__backup_path = os.path.join(folder, "backup")

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
