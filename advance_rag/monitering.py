"""
Monitoring and logging for production
Structured logging, metrics, and alerts
"""

import logging
import json
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage