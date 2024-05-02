from carPlateDetection.logger import logging
from carPlateDetection.exception import AppException
import sys

try: 
    a = 3 / "s"
    
except Exception as e:
    raise AppException(e, sys)