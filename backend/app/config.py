from dotenv import load_dotenv
import os

# Load environment variables from the .env file inside the app folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Get the DATABASE_URL from the environment
DATABASE_URL = os.getenv("DATABASE_URL")


