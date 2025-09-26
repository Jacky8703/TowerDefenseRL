import datetime
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import json
import argparse
import os

SERVER_URL = "http://localhost:3000"
SET_MAP_ENDPOINT = f"{SERVER_URL}/set-map"
RESET_ENDPOINT = f"{SERVER_URL}/reset"
STEP_ENDPOINT = f"{SERVER_URL}/step"
RENDER_ENDPOINT = f"{SERVER_URL}/render"
PLAYBACK_SPEED_MS = 10  # delay between frames in milliseconds
DEFAULT_MAP_WAYPOINTS = [
    { "x": 75, "y": 25 },
    { "x": 75, "y": 325 },
    { "x": 225, "y": 325 },
    { "x": 225, "y": 475 },
    { "x": 375, "y": 475 },
    { "x": 375, "y": 125 },
    { "x": 825, "y": 125 },
    { "x": 825, "y": 275 },
    { "x": 525, "y": 275 },
    { "x": 525, "y": 525 },
    { "x": 725, "y": 525 },
    { "x": 725, "y": 575 }
]

def main(actions_file, save_frames, load_dir):
    """
    Replays a game by either collecting frames from a server or loading them from a directory.
    """
    frames = []
    target_wave = 0

    # --- Determine the mode: Load from disk or collect from server ---
    if load_dir:
        # --- Load frames from a directory ---
        print(f"--- Loading frames from directory: {load_dir} ---")
        if not os.path.isdir(load_dir):
            print(f"Error: Directory not found at '{load_dir}'")
            return
        
        try:
            # Get all image files and sort them to ensure correct order
            image_files = sorted([f for f in os.listdir(load_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if len(image_files) == 0:
                print(f"Error: No image files found in '{load_dir}'")
                return

            start_time = datetime.datetime.now()
            for i, filename in enumerate(image_files):
                print(f"Loading frame {i + 1}/{len(image_files)}...", end='\r')
                frame_path = os.path.join(load_dir, filename)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
            print(f"Successfully loaded {len(frames)} frames in {(datetime.datetime.now() - start_time).seconds} seconds.")
        except Exception as e:
            print(f"An error occurred while loading frames: {e}")
            return

    elif actions_file:
        # --- Collect frames from the server ---
        print(f"--- Collecting frames from server using: {actions_file} ---")
        try:
            with open(actions_file, 'r') as f:
                data = json.load(f)
                actions = data["actions"]
                target_wave = data["wave_number"]
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return

        save_dir = None
        if save_frames:
            # Get the directory where the actions file is located
            base_dir = os.path.dirname(os.path.abspath(actions_file))
            # Create the path for the 'best_frames' folder
            save_dir = os.path.join(base_dir, "best_frames")

        try:
            requests.post(SET_MAP_ENDPOINT, json=DEFAULT_MAP_WAYPOINTS).raise_for_status()
            requests.post(RESET_ENDPOINT).raise_for_status()

            start_time = datetime.datetime.now()
            for i, action in enumerate(actions):
                print(f"Processing action {i + 1}/{len(actions)}...", end='\r')
                response = requests.post(STEP_ENDPOINT, json=action)
                if response.status_code == 400:
                    error_msg = response.json()["message"]
                    print(f"\nAction {i + 1} was invalid: {error_msg}. Continuing.")
                    continue
                response.raise_for_status()
                
                render_response = requests.get(RENDER_ENDPOINT)
                if render_response.status_code == 400:
                    error_msg = render_response.json()["message"]
                    print(f"\nError rendering frame at action {i + 1}: {error_msg}. Stopping.")
                    break
                render_response.raise_for_status()
                
                image_bytes = BytesIO(render_response.content)
                image = Image.open(image_bytes).convert("RGB")
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                frames.append(frame)
            
            print(f"\nFrame collection complete. Collected {len(frames)} frames in {(datetime.datetime.now() - start_time).seconds} seconds.")
            # save only after collecting all frames
            if save_dir:
                print(f"Frames will be saved to: {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    print(f"Saving frame {i + 1}/{len(frames)}...", end='\r')
                    frame_filename = os.path.join(save_dir, f"frame_{i:04d}.png")
                    cv2.imwrite(frame_filename, frame)

        except Exception as e:
            print(f"\nAn error occurred during frame collection: {e}")
            return
    else:
        print("Error: You must provide an actions file (--actions-file) or a directory to load from (--load-dir).")
        return

    # --- Replay collected frames ---
    if not frames:
        print("No frames to display. Exiting.")
        return

    print("\n--- Starting replay ---")
    if target_wave != 0:
        print(f"Episode reached wave: {target_wave}")
    print("Press 'q' to quit.")

    while True:
        for frame in frames:
            cv2.imshow("Game Replay", frame)
            if cv2.waitKey(PLAYBACK_SPEED_MS) & 0xFF == ord('q'):
                print("Playback stopped by user.")
                return
        
        print("\nReplay finished. Press 'q' to quit or 'r' to replay.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Exiting.")
                return
            elif key == ord('r'):
                print("Replaying...")
                break

def parse_arguments():
    parser = argparse.ArgumentParser(description="Replay a Tower Defense game from a saved actions file or a directory of frames.")
    parser.add_argument("--actions-file", help="Path to the JSON actions file.")
    parser.add_argument("--save-frames", action="store_true", help="Optional. Save frames to a 'best_frames' directory next to the actions file.")
    parser.add_argument("--load-dir", help="Optional. Directory to load frames.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(args.actions_file, args.save_frames, args.load_dir)
    finally:
        cv2.destroyAllWindows()