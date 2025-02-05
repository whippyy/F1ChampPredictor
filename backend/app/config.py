from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Database URL from the environment
DATABASE_URL = os.getenv("DATABASE_URL")

    


