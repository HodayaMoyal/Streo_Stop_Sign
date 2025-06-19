import numpy as np
import math
from flask_cors import CORS
from flask import Flask, request, jsonify
import time
from ultralytics import YOLO
from geopy.distance import geodesic 
import os
from datetime import datetime
import pyautogui
import cv2
import yaml
import torch
from collections import defaultdict
from glob import glob
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import shutil
import configparser

# https://colab.research.google.com/drive/1yAJCZfmcwo63WfNEJUvknU_YMb_d4k8p#scrollTo=6mrqi2YShUe4