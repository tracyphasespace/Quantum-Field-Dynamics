from datetime import datetime, timezone

try:
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    print(f"SUCCESS: Timestamp generated: {timestamp}")
except NameError as e:
    print(f"NameError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
