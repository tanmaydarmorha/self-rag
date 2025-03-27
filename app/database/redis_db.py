import os
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis connection details from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Create Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

def get_redis_connection():
    """Get Redis connection."""
    return redis_client

# Check connection
def check_redis_connection():
    """Check if Redis connection is working."""
    try:
        return redis_client.ping()
    except Exception as e:
        print(f"Redis connection error: {e}")
        return False
