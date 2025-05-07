from pynput import keyboard
import pandas as pd
import time
import os

user_id = input("Enter your user ID: ")
fixed_text = ".tie5Roanl"  # Change to match the original dataset password
print(f"\n📝 Please type this password exactly:\n    '{fixed_text}'")
print("⏳ Starting in 3 seconds. Get ready...")
time.sleep(3)
print("⌨️  Start typing now. Press ENTER when done.\n")

events = []

def on_press(key):
    try:
        events.append({'event': 'press', 'key': key.char, 'time': time.time()})
    except AttributeError:
        pass

def on_release(key):
    try:
        events.append({'event': 'release', 'key': key.char, 'time': time.time()})
    except AttributeError:
        pass
    if key == keyboard.Key.enter:
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Extract timing features
key_sequence = list(fixed_text)
press_times = {}
release_times = {}
H = {}   # Hold times
DD = {}  # Down-Down times
UD = {}  # Up-Down times

for event in events:
    k = event['key']
    if event['event'] == 'press' and k not in press_times:
        press_times[k] = event['time']
    elif event['event'] == 'release' and k not in release_times:
        release_times[k] = event['time']

# Compute hold times
for k in key_sequence:
    if k in press_times and k in release_times:
        H[k] = release_times[k] - press_times[k]

# Compute DD and UD times
DD_pairs = {}
UD_pairs = {}
for i in range(len(key_sequence) - 1):
    k1 = key_sequence[i]
    k2 = key_sequence[i + 1]
    if k1 in press_times and k2 in press_times:
        DD_pairs[f'{k1}.{k2}'] = press_times[k2] - press_times[k1]
    if k1 in release_times and k2 in press_times:
        UD_pairs[f'{k1}.{k2}'] = press_times[k2] - release_times[k1]

# Combine all into a single row
row = {
    'subject': user_id,
    'sessionIndex': 1,
    'rep': 1
}
for k in key_sequence:
    row[f'H.{k}'] = H.get(k, 0)
for pair, value in DD_pairs.items():
    row[f'DD.{pair}'] = value
for pair, value in UD_pairs.items():
    row[f'UD.{pair}'] = value

# Save final formatted row
df = pd.DataFrame([row])
os.makedirs('data', exist_ok=True)
df.to_csv(f'data/{user_id}_raw.csv', index=False)
print(f"\n✅ Processed data saved to data/{user_id}_raw.csv")
