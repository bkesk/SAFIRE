# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# This file includes portions derived from work licensed under the
# University of Illinois/NCSA Open Source License. See the NOTICE file
# and LICENSES/NCSA.txt for details.

import logging

# Create a new logging level
# see docs for details: https://docs.python.org/3/library/logging.html#levels
DEV = 15 # this is between INFO and WARNING
logging.addLevelName(DEV, 'DEV')
logging.DEV = DEV

def dev(self, message, *args, **kwargs):
    if self.isEnabledFor(DEV):
        # Note: stacklevel 2 is necessary to properly resolve the caller; 
        #           othrwise the log will point to this function (i.e. `dev()`), 
        #           instead of the calling function.
        self._log(DEV, message, args, stacklevel=2, **kwargs)

logging.Logger.dev = dev

# a possibly useful configuration for debbuging / development:
# logging.basicConfig('%(levelname)s: %(filename)s %(funcName)s :%(message)s', level=logging.DEV)
# NOTE: the line above is a global configuration, and will affect all loggers

