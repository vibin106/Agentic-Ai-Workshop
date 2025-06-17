# config.py

import os

# Store your key here (securely for dev only)
GEMINI_API_KEY = "AIzaSyDmW77hkxj5xFtGtu3GNmI9Oik59kvl9qA"  # <-- Replace this once

# Set environment variable when imported
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Model to use
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
