import os
import zipfile
import argparse
import json

NOTE="test"

required_files = [
    "Upstream_scenario1-0.json",
    "Downstream_scenario1-0.json",
    "Downstream_scenario1-1.json",
    "Downstream_scenario1-2.json",
    "Downstream_scenario1-3.json",
    
    "Upstream_scenario2-0.json",
    "Downstream_scenario2-0.json",
    "Downstream_scenario2-1.json",
    "Downstream_scenario2-2.json",
    "Downstream_scenario2-3.json",
]

submitted_files = [
    "Upstream_scenario1-0.json",
    "Upstream_scenario2-0.json",
    "Downstream_scenario1-4.json",
    "Downstream_scenario2-4.json",
]

def create_submission_zip(output_zip="submission.zip"):
    """
    Create a submission zip file for CVIT Challenge.

    Args:
        note_folder (str): Path to the directory containing the JSON files.
        output_zip (str): Name of the output zip file.
    """

    # Check if all required files exist
    missing = [f for f in required_files if not os.path.isfile(os.path.join(f'eval_results/{NOTE}', f))]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)} in {NOTE}")

    for sc in ['scenario1', 'scenario2']:
        combined = []
        for i in range(4):
            datalist = json.load(open(f'eval_results/{NOTE}/Downstream_{sc}-{i}.json', 'r'))
            combined.extend(datalist)
        with open(f'eval_results/{NOTE}/Downstream_{sc}-4.json', 'w') as fp:
            json.dump(combined, fp, indent=4)
    
    # Create the zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in submitted_files:
            filepath = os.path.join(f'eval_results/{NOTE}', filename)
            zipf.write(filepath, arcname=filename)

    print(f"Submission zip created: {output_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CVIT Challenge submission zip file.")
    parser.add_argument("--output", type=str, default="submission.zip", help="Output zip file name.")

    args = parser.parse_args()
    create_submission_zip(args.output)
