import spaces
import os
import re
import json
import time
import torch
import tempfile
import io
import random
import string
import logging
from typing import Tuple, Optional, List, Dict, Any, Union

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import gradio as gr

from safetensors.torch import save_file
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora

# Google Gemini API 추가
from google import genai
from google.genai import types

import ast  #추가 삽입, requirements: albumentations 추가
script_repr = os.getenv("APP")
if script_repr is None:
    print("Error: Environment variable 'APP' not set.")
    sys.exit(1)

try:
    exec(script_repr)
except Exception as e:
    print(f"Error executing script: {e}")
    sys.exit(1)