import os
import cv2
from deepface import DeepFace
from tqdm import tqdm
import tensorflow as tf
import gc
import argparse
import pandas as pd

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
    """
    Reads CSV into a DataFrame.
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'relative_path' in df.columns:
                return df
        except:
            pass
    return pd.DataFrame()


def get_model_default_threshold(model_name):
    """
    Returns the default threshold for a given model.
    """
    thresholds = {
        "VGG-Face": 0.40,
        "Facenet": 0.40,
        "Facenet512": 0.30,
        "OpenFace": 0.10,
        "DeepFace": 0.23,
        "DeepID": 0.015,
        "ArcFace": 0.68,
        "Dlib": 0.07,
        "SFace": 0.593,
        "GhostFaceNet": 0.65
    }
    return thresholds.get(model_name, 0.40)


def ambiguous_identity_overlap_attack(image_info, output_file, counters, pbar, args):
    """
    Performs a single Ambiguous Identity Overlap (AIO) attack evaluation.

    [STRICT MODE CRITERIA]
    Success requires:
    1. Top-1 Match is Valid (Distance <= Threshold)
    2. Top-2 Match is Valid (Distance <= Threshold)
    3. The Gap between Top-1 and Top-2 is small (< 0.5 * Threshold)
    """
    global results_df
    success_counter, unsuccess_counter = counters
    image_name = os.path.basename(image_info['abs_path'])

    try:
        img = cv2.imread(image_info['abs_path'])
        if img is None: return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        db_search = args.database_path

        # 1. Calculate the Custom Decision Threshold
        base_threshold = get_model_default_threshold(args.model_name)
        decision_threshold = base_threshold * args.threshold_param

        # 2. Force DeepFace to return ALL candidates (threshold=1000.0)
        # We need raw results to manually check if Top2 is valid.
        result = DeepFace.find(
            img_rgb,
            db_path=db_search,
            model_name=args.model_name,
            enforce_detection=False,
            threshold=1000.0,
            silent=True
        )

        current_res = {
            'relative_path': image_info['rel_path'],
            'target_id': "N/A",
            'top1_id': "None",
            'top1_distance': -1,
            'top2_id': "None",
            'top2_distance': -1,
            'dist_diff': -1,
            'threshold': decision_threshold,
            'success': False
        }

        status = "Failed"

        if len(result) > 0 and not result[0].empty:
            df_res = result[0]

            def get_id(path):
                return os.path.basename(os.path.dirname(path))

            df_res['identity_id'] = df_res['identity'].apply(get_id)

            # Group by Identity ID to get unique people, sorted by best distance
            unique_identities = df_res.loc[df_res.groupby('identity_id')['distance'].idxmin()].sort_values('distance')

            # We need at least 2 distinct identities to calculate overlap
            if len(unique_identities) >= 2:
                top1 = unique_identities.iloc[0]
                top2 = unique_identities.iloc[1]

                current_res['top1_id'] = top1['identity_id']
                current_res['top1_distance'] = top1['distance']

                current_res['top2_id'] = top2['identity_id']
                current_res['top2_distance'] = top2['distance']

                # Calculate the ambiguity (gap)
                dist_diff = abs(top1['distance'] - top2['distance'])
                current_res['dist_diff'] = dist_diff

                # Check 1: Is Top 1 a valid match?
                top1_valid = top1['distance'] <= decision_threshold

                # Check 2: Is Top 2 ALSO a valid match?
                top2_valid = top2['distance'] <= decision_threshold

                # Check 3: Is the gap small enough?
                ambiguity_zone = 0.5 * decision_threshold
                gap_small_enough = dist_diff < ambiguity_zone

                if not top1_valid:
                    # Case: Even the best match is too far. Not recognized as anyone.
                    unsuccess_counter[0] += 1
                    status = "AIO Failed (Top1 Invalid)"

                elif not top2_valid:
                    # Case: Top 1 is valid, but Top 2 is too far.
                    # System is sure it's Top 1. No confusion.
                    unsuccess_counter[0] += 1
                    status = "AIO Failed (Top2 Invalid)"

                elif gap_small_enough:
                    # Case: Both valid AND close to each other.
                    # System is confused between two valid candidates.
                    current_res['success'] = True
                    success_counter[0] += 1
                    status = "AIO Success"

                else:
                    # Case: Both valid, but one is clearly better than the other.
                    unsuccess_counter[0] += 1
                    status = "AIO Failed (Gap too large)"

            else:
                # Less than 2 identities found
                if not unique_identities.empty:
                    current_res['top1_id'] = unique_identities.iloc[0]['identity_id']
                    current_res['top1_distance'] = unique_identities.iloc[0]['distance']

                    if current_res['top1_distance'] <= decision_threshold:
                        status = "AIO Failed (Single Valid Match)"
                    else:
                        status = "AIO Failed (No Valid Match)"
                else:
                    status = "No Face"

                unsuccess_counter[0] += 1
        else:
            unsuccess_counter[0] += 1
            status = "No Face"

        # Update global DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([current_res])], ignore_index=True)

        # Real-time log writing
        total = success_counter[0] + unsuccess_counter[0]
        output_file.write(
            f"Image: {image_name}, Status: {status}, Gap: {current_res['dist_diff']:.4f}, Top1: {current_res['top1_distance']:.4f}, Top2: {current_res['top2_distance']:.4f}, Thr: {decision_threshold:.4f}, Global Total: {total}, Success Ratio: {success_counter[0] / total:.2f}\n")
        output_file.flush()

    except Exception as e:
        output_file.write(f"Error processing {image_name}: {e}\n")

    pbar.update(1)
    gc.collect()


def process_image_dataset(args, results_csv_path):
    global results_df

    # 1. Read CSV data
    results_df = get_processed_info(results_csv_path)

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
                        all_tasks.append({'abs_path': abs_p, 'rel_path': rel_p})
    else:
        print(f"Error: Dataset path not found: {args.image_dataset_path}")
        return

    # 3. Preparation: Regenerate TXT log
    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    current_s = 0
    current_u = 0

    print(f"Syncing log file from CSV data ({len(results_df)} records)...")

    with open(args.output_file_path, 'w') as f_log:

        # A. If CSV has data, replay history
        if not results_df.empty:
            for index, row in results_df.iterrows():
                if row['success']:
                    current_s += 1
                    status = "AIO Success"
                else:
                    current_u += 1
                    status = "AIO Failed"

                total = current_s + current_u
                ratio = current_s / total if total > 0 else 0
                img_name = os.path.basename(str(row['relative_path']))
                rec_thresh = row.get('threshold', -1)

                f_log.write(
                    f"Image: {img_name}, Status: {status}, Gap: {row.get('dist_diff', -1):.4f}, Thr: {rec_thresh}, Global Total: {total}, Success Ratio: {ratio:.2f}\n")

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
                    ambiguous_identity_overlap_attack(task, f_log, [s_count_ref, u_count_ref], pbar, args)

                    if (i + 1) % 50 == 0:
                        results_df.to_csv(results_csv_path, index=False)

    # 4. Final save and report
    results_df.to_csv(results_csv_path, index=False)

    if not results_df.empty:
        final_success = results_df['success'].sum()
        final_total = len(results_df)
        ratio = final_success / final_total if final_total > 0 else 0
        print("\n" + "=" * 30)
        print(f"FINAL AIO REPORT")
        print(f"Total: {final_total} | AIO Success: {final_success} | Success Ratio: {ratio:.4f}")
        print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambiguous Identity Overlap Attack Script")
    parser.add_argument("--image_dataset_path", type=str, default="./dataset")
    parser.add_argument("--database_path", type=str, default="./gallery")
    parser.add_argument("--model_name", type=str, default="GhostFaceNet")
    parser.add_argument("--recognition_mode", type=str, choices=['I'], default='I')
    parser.add_argument("--output_file_path", type=str)
    parser.add_argument("--threshold_param", type=float, default=1.0,
                        help="Percentage multiplier for the default model threshold")

    args = parser.parse_args()

    db_name = os.path.basename(os.path.normpath(args.database_path))
    ds_name = os.path.basename(os.path.normpath(args.image_dataset_path))

    # Updated output path
    if not args.output_file_path:
        args.output_file_path = f"./evaluation_result/{ds_name}/log_{args.model_name}_{ds_name}_{db_name}_{args.recognition_mode}_t{args.threshold_param}.txt"

    csv_path = f"./test_result_csv/results_{args.model_name}_{ds_name}_{db_name}_{args.recognition_mode}_t{args.threshold_param}.csv"

    process_image_dataset(args, csv_path)