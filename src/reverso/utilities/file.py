import os


def getListOfFiles(dirName: str):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def findFileInDirectory(dirName: str, patterns: list, itemCount: int = None):
    assert dirName is not None
    if isinstance(patterns, str):
        patterns = [patterns]

    files = getListOfFiles(dirName)
    resultFiles = []
    for file in files:
        for pattern in patterns:
            if file.find(pattern) >= 0:
                resultFiles.append(file)
                break
    if len(resultFiles) == 0:
        return None
    elif itemCount == 1:
        return resultFiles[0]
    elif itemCount is None or itemCount < 0 or itemCount >= len(resultFiles):
        return resultFiles
    else:
        return resultFiles[0:itemCount]


def getSizeOfDirectory(dirName: str):
    sizes = [os.path.getsize(f) for f in getListOfFiles(dirName)]
    return sum(sizes)
