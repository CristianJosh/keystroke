import pandas as pd
import joblib
import time
from pynput import keyboard

# Constants
fixed_password = ".tie5Roanl"
key_sequence = list(fixed_password)
press_times = {}
release_times = {}

print(f"📝 Type the password: {fixed_password}")
print("⌨️  Press ESC after finishing.\n")

# Capture timing events
def on_press(key):
    try:
        k = key.char
        if k not in press_times:
            press_times[k] = time.time()
            print(f"Pressed: {k}")
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char
        if k not in release_times:
            release_times[k] = time.time()
            print(f"Released: {k}")
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        return False

# Start listening
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Prepare features DSL-style
row = {}

# Hold times (H.<key>)
for k in key_sequence:
    row[f'H.{k}'] = release_times.get(k, 0) - press_times.get(k, 0)

# Down-Down times (DD.<k1>.<k2>)
for i in range(len(key_sequence) - 1):
    k1, k2 = key_sequence[i], key_sequence[i + 1]
    row[f'DD.{k1}.{k2}'] = press_times.get(k2, 0) - press_times.get(k1, 0)

# Up-Down times (UD.<k1>.<k2>)
for i in range(len(key_sequence) - 1):
    k1, k2 = key_sequence[i], key_sequence[i + 1]
    row[f'UD.{k1}.{k2}'] = press_times.get(k2, 0) - release_times.get(k1, 0)

# Create dataframe for prediction
feature_vector = pd.DataFrame([row])

# Load trained model
model = joblib.load('models/keystroke_model.pkl')

# Predict user
predicted_user = model.predict(feature_vector)[0]
print(f"\n✅ Predicted User: {predicted_user}")
