"""
Configuration for Astar Island challenge.
Replace ACCESS_TOKEN with your JWT from app.ainm.no cookies.
"""

BASE_URL = "https://api.ainm.no"
ACCESS_TOKEN = "your-jwt-token-here"

# Prediction constants
PROB_FLOOR = 0.01  # Minimum probability to avoid KL divergence blowup
NUM_CLASSES = 6
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Terrain code -> class index mapping
TERRAIN_TO_CLASS = {
    0: 0,   # Empty -> Empty
    10: 0,  # Ocean -> Empty
    11: 0,  # Plains -> Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
}
