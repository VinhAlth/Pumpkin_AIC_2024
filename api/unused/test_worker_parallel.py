import sys
import os
import subprocess
import flask
from flask import request





app = flask.Flask("API Text Search")
app.config["DEBUG"] = False

print(count_other_processes_using_current_file())