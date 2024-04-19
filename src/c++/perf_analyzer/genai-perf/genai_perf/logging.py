# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import logging.config
from math import log
from pathlib import Path


def init_logging() -> None:
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M",
            },
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
        },
        "loggers": {
            "": {  # root logger - avoid using
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.parser": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "genai_perf.wrapper": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


def add_file_logger(log_file: Path) -> None:
    print(log_file)
    # Incremental configuration to add a file handler
    add_file_handler = {
        "version": 1,
        "incremental": True,
        # "handlers": {
        "file": {  # Adding a new file handler
            "level": "WARNING",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(log_file),
            "mode": "a",
            "encoding": "utf-8",
            "force": True,
        },
        # },
        "__main__": {
            "handlers": ["console", "file"],  # Now using both console and file handlers
        },
        "genai_perf.parser": {
            "handlers": ["console", "file"],
        },
        "genai_perf.wrapper": {
            "handlers": ["console", "file"],
        },
    }
    # update_loggers_with_file = {
    #     "version": 1,
    #     "incremental": True,
    #     "__main__": {
    #         "handlers": ["console", "file"],  # Now using both console and file handlers
    #     },
    #     "genai_perf.parser": {
    #         "handlers": ["console", "file"],
    #     },
    #     "genai_perf.wrapper": {
    #         "handlers": ["console", "file"],
    #     },
    # }

    # Apply the incremental configuration
    logging.config.dictConfig(add_file_handler)
    # logging.config.dictConfig(update_loggers_with_file)


def getLogger(name):
    return logging.getLogger(name)
