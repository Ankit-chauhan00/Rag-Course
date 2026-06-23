"""
just for getting the name of the embedding models

"""

from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

for model in client.models.list():
    print(model.name)
