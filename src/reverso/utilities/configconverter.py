from configparser import ConfigParser
import os
import sys


def readConfiguration(filename: str) -> dict:
    config = ConfigParser()
    if filename != None and os.path.exists(filename):
        config.read(filename)
        return convertConfigToDic(config)
    else:
        return {}


def convertConfigToDic(config: ConfigParser) -> dict:

    sections_dict = {}

    # get all defaults
    defaults = config.defaults()
    temp_dict = {}
    for key in defaults.keys():
        temp_dict[key] = defaults[key]

    sections_dict['default'] = temp_dict

    # get sections and iterate over each
    sections = config.sections()

    for section in sections:
        options = config.options(section)
        temp_dict = {}
        for option in options:
            temp_dict[option] = config.get(section, option)

        sections_dict[section] = temp_dict

    return sections_dict
