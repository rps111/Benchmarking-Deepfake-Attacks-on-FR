import os
import cv2
from deepface import DeepFace
from tqdm import tqdm
import tensorflow as tf
import gc
import argparse
import pandas as pd
import numpy as np

# --- Environment Configuration ---
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"Error setting memory growth for GPU: {e}")

# Global variable
results_df = pd.DataFrame()


def get_processed_info(csv_path):
    """Reads CSV into a DataFrame."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'relative_path' in df.columns:
                return df
        except:
            pass
    return pd.DataFrame()


def get_model_default_threshold(model_name, metric='cosine'):
    """
    Returns the default threshold for a given model and metric.
    Based on DeepFace library defaults.
    """
    # Standard thresholds for Cosine similarity in DeepFace
    thresholds = {
        "VGG-Face": 0.68,
        "Facenet": 0.40,
        "ArcFace": 0.68,
        "Dlib": 0.07,
        "SFace": 0.593,
        "GhostFaceNet": 0.65
    }

    # Simple fallback if model name case doesn't match or is unknown
    # You might want to adjust these based on the metric if you change from cosine
    return thresholds.get(model_name, 0.40)


def direct_impersonation_attack(image_info, target_id_path, output_file, counters, pbar, args):
    """Performs a single attack evaluation with custom threshold logic."""
    global results_df
    success_counter, unsuccess_counter = counters
    image_name = os.path.basename(image_info['abs_path'])

    try:
        img = cv2.imread(image_info['abs_path'])
        if img is None: return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        db_search = target_id_path if args.recognition_mode == 'V' else args.database_path

        # 1. Calculate the Custom Decision Threshold
        # Get standard default for this model
        base_threshold = get_model_default_threshold(args.model_name, metric='cosine')
        # Apply user scaling (e.g., 1.1 for +10% tolerance, 0.9 for stricter)
        decision_threshold = base_threshold * args.threshold_param

        # 2. Force DeepFace to return results by setting a huge threshold for the search
        # We use 1000 (or effectively infinite) so it returns the nearest neighbor regardless of distance
        result = DeepFace.find(
            img_rgb,
            db_path=db_search,
            model_name=args.model_name,
            enforce_detection=False,
            threshold=1000.0,  # Huge threshold to force return
            silent=True
        )

        current_res = {
            'relative_path': image_info['rel_path'],
            'target_id': image_info['target_id'],
            'matched_id': "None",
            'distance': -1,  # Default
            'threshold': decision_threshold,  # Log the calculated threshold we used
            'success': False
        }

        # 3. Manual Evaluation
        if len(result) > 0 and not result[0].empty:
            first_row = result[0].iloc[0]
            matched_id = os.path.basename(os.path.dirname(first_row['identity']))
            distance = first_row['distance']

            current_res['matched_id'] = matched_id
            current_res['distance'] = distance

            # --- CUSTOM SUCCESS LOGIC ---
            # Check 1: Is the distance within our custom scaled threshold?
            # Check 2: Is the ID correct?
            if distance <= decision_threshold and matched_id == image_info['target_id']:
                current_res['success'] = True
                success_counter[0] += 1
                status = "Success"
            else:
                # Even if ID matches, if distance is too high, it's a fail.
                # Even if distance is low, if ID is wrong, it's a fail.
                unsuccess_counter[0] += 1
                status = "Failed"
        else:
            # Should rarely happen with threshold=1000 unless no faces detected at all
            unsuccess_counter[0] += 1
            status = "No Matched Face"

        # Update global DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([current_res])], ignore_index=True)

        # Real-time log writing
        total = success_counter[0] + unsuccess_counter[0]
        output_file.write(
            f"Image: {image_name}, Status: {status}, Dist: {current_res['distance']:.4f}, Thr: {decision_threshold:.4f}, Global Total: {total}, Ratio: {success_counter[0] / total:.2f}\n")
        output_file.flush()

    except Exception as e:
        output_file.write(f"Error processing {image_name}: {e}\n")
        print(f"Error: {e}")

    pbar.update(1)
    gc.collect()


def process_image_dataset(args, results_csv_path):
    global results_df

    # 1. Read CSV data
    results_df = get_processed_info(results_csv_path)

    # Get set of processed paths
    processed_paths = set()
    if not results_df.empty:
        processed_paths = set(results_df['relative_path'].unique())

    # 2. Prepare all pending tasks
    all_tasks = []
    if os.path.exists(args.image_dataset_path):
        for root, _, files in os.walk(args.image_dataset_path):
            for name in files:
                if name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    abs_p = os.path.join(root, name)
                    rel_p = os.path.relpath(abs_p, args.image_dataset_path)

                    if rel_p not in processed_paths:
                        t_id = os.path.basename(root).split('_')[0]
                        all_tasks.append({'abs_path': abs_p, 'rel_path': rel_p, 'target_id': t_id})
    else:
        print(f"Error: Dataset path not found: {args.image_dataset_path}")
        return

    # 3. Preparation: Regenerate TXT log
    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    current_s = 0
    current_u = 0

    print(f"Syncing log file from CSV data ({len(results_df)} records)...")

    # Open TXT in 'w' mode to rewrite history
    with open(args.output_file_path, 'w') as f_log:

        # A. If CSV has data, replay history
        if not results_df.empty:
            for index, row in results_df.iterrows():
                if row['success']:
                    current_s += 1
                    status = "Success"
                else:
                    current_u += 1
                    dist_val = row.get('distance', -1)
                    if dist_val == -1 or pd.isna(dist_val):
                        status = "No Matched Face"
                    else:
                        status = "Failed"

                total = current_s + current_u
                ratio = current_s / total if total > 0 else 0
                img_name = os.path.basename(str(row['relative_path']))

                # Try to get threshold from CSV if it exists, else placeholder
                rec_thresh = row.get('threshold', -1)

                f_log.write(
                    f"Image: {img_name}, Status: {status}, Dist: {row.get('distance', -1):.4f}, Thr: {rec_thresh}, Global Total: {total}, Ratio: {ratio:.2f}\n")

            f_log.flush()
            print(f"History restored to TXT. Resuming task...")

        # B. Start running new tasks
        s_count_ref = [current_s]
        u_count_ref = [current_u]

        if not all_tasks:
            print("No new images to process.")
        else:
            with tqdm(total=len(all_tasks), desc="Evaluating", unit="img") as pbar:
                for i, task in enumerate(all_tasks):
                    target_path = os.path.join(args.database_path, task['target_id'])

                    if not os.path.exists(target_path):
                        # If target not found in DB, skip
                        pbar.update(1)
                        continue

                    direct_impersonation_attack(task, target_path, f_log, [s_count_ref, u_count_ref], pbar, args)

                    if (i + 1) % 50 == 0:
                        results_df.to_csv(results_csv_path, index=False)

    # 4. Final save and report
    results_df.to_csv(results_csv_path, index=False)

    if not results_df.empty:
        final_success = results_df['success'].sum()
        final_total = len(results_df)
        ratio = final_success / final_total if final_total > 0 else 0
        print("\n" + "=" * 30)
        print(f"FINAL REPORT")
        print(f"Total: {final_total} | Success: {final_success} | Ratio: {ratio:.4f}")
        print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dataset_path", type=str, default="./dataset")
    parser.add_argument("--database_path", type=str, default="./gallery")
    parser.add_argument("--model_name", type=str, default="Facenet")
    parser.add_argument("--recognition_mode", type=str, choices=['V', 'I'], default='V')
    parser.add_argument("--output_file_path", type=str)
    parser.add_argument("--threshold_param", type=float, default=1.0,
                        help="Percentage multiplier for the default model threshold (e.g., 1.1 for 110%)")

    args = parser.parse_args()
    db_name = os.path.basename(os.path.normpath(args.database_path))
    ds_name = os.path.basename(os.path.normpath(args.image_dataset_path))

    # Updated output path naming to include threshold param so logs don't mix up
    if not args.output_file_path:
        args.output_file_path = f"./evaluation_result/{ds_name}/log_{args.model_name}_{ds_name}_{db_name}_{args.recognition_mode}_t{args.threshold_param}.txt"

    csv_path = f"./test_result_csv/results_{args.model_name}_{ds_name}_{db_name}_{args.recognition_mode}_t{args.threshold_param}.csv"

    process_image_dataset(args, csv_path)