import os
import shutil
import uuid


class TempHandler:
    _folder: str
    _sessionId_start = 0

    def __init__(self, folder: str, reset=False):
        self._folder = folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        elif reset:
            self.removeAllSession()

    def _getSessionPath(self, sessionId: int):
        return os.path.join(self._folder, "dtemp_" + str(sessionId))

    def createSession(self) -> int:
        is_unique = False
        sessionId = self._sessionId_start
        folder_path = None
        while not is_unique:
            sessionId = sessionId + 1
            folder_path = self._getSessionPath(sessionId)
            is_unique = not os.path.exists(folder_path)
        os.mkdir(folder_path)

        self._sessionId_start = sessionId
        return sessionId

    def generatePathname(self, sessionId: int):
        is_unique = False
        pathname = None
        folder_path = self._getSessionPath(sessionId)
        while not is_unique:
            name = str(uuid.uuid4())
            pathname = os.path.join(folder_path, name)
            is_unique = not os.path.exists(pathname)

        os.mkdir(pathname)
        return pathname

    def generateFilename(self, sessionId: int, fileType: str = None) -> str:
        if fileType is None:
            fileType = ""
        elif fileType[0] != ".":
            fileType = "." + fileType

        is_unique = False
        filename = None
        folder_path = self._getSessionPath(sessionId)
        while not is_unique:
            name = str(uuid.uuid4())
            filename = os.path.join(folder_path, name + fileType)
            is_unique = not os.path.exists(filename)

        return filename

    def removeSession(self, sessionId):
        folder_path = self._getSessionPath(sessionId)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    def removeAllSession(self):
        shutil.rmtree(self._folder)
        os.mkdir(self._folder)
