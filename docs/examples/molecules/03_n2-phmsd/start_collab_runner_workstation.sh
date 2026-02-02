#!/usr/bin/env bash

module load safire/0.3

jupyter-lab . \
	    --ip=0.0.0.0 \
	    --ServerApp.port=9999 \
	    --ServerApp.port_retries=0 \
	    --no-browser

### Then, on a different machine, forward a port:
#
#   `$ ssh -L127.0.0.1:8888:ccqlin[###]:8888 flatiron`
#
#  WARNING!!! BE SURE TO CLOSE THIS WEHN YOU ARE DONE!!!
#
#################################################
