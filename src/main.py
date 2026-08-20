from dotenv import load_dotenv

from .app import create_fast_api_app

load_dotenv()
app = create_fast_api_app()
