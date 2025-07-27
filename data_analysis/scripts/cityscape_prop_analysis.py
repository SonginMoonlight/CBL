#!/usr/bin/env python3
"""
Cityscapes Object Analyzer
Analyzes JSON annotation files to find instances of cars, people, and buildings.
Calculates the proportion of each instance relative to total image area.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Target object names based on Cityscapes label conventions
# Cityscapes uses more specific vehicle categories and person types
CAR_NAMES = {'car'}
PERSON_NAMES = {'person'}
BUILDING_NAMES = {'building'}


def calc_area(polygon: List[List[float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2


def parse_json(json_path: str) -> Tuple[Dict[str, List[float]], Tuple[int, int]]:
    """Parse Cityscapes JSON file to extract object instances and areas."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}, (0, 0)

    # Initialize result containers
    cars = []
    people = []
    buildings = []

    # Get image dimensions - Cityscapes stores them at root level
    img_width = data.get('imgWidth', 0)
    img_height = data.get('imgHeight', 0)

    total_area = img_width * img_height
    if total_area == 0:
        return {}, (0, 0)

    # Parse objects - Cityscapes stores them in 'objects' array
    objects = data.get('objects', [])

    for obj in objects:
        # Get object label - Cityscapes uses 'label' field
        obj_label = obj.get('label', '').strip().lower()

        # Get polygon coordinates - Cityscapes stores as direct coordinate pairs
        polygon = obj.get('polygon', [])

        if not polygon or len(polygon) < 3:
            continue

        # Calculate area proportion
        area = calc_area(polygon)
        proportion = area / total_area

        # Classify object based on Cityscapes labels
        if obj_label in CAR_NAMES:
            cars.append(proportion)
        elif obj_label in PERSON_NAMES:
            people.append(proportion)
        elif obj_label in BUILDING_NAMES:
            buildings.append(proportion)

    return {
               'car': cars,
               'person': people,
               'building': buildings
           }, (img_width, img_height)


def analyze_dataset(root_path: str) -> List[Dict]:
    """Traverse Cityscapes dataset and collect statistics for all JSON files."""
    results = []
    root = Path(root_path)

    # Find all JSON polygon files
    # Cityscapes typically stores annotations in *_polygons.json files
    json_files = list(root.rglob('*_polygons.json'))

    if not json_files:
        # Fallback to any JSON files if specific pattern not found
        json_files = [p for p in root.rglob('*.json')
                      if 'polygons' in p.name.lower()]

    files_with_targets = 0
    unique_labels = set()

    # Process files with progress bar
    with tqdm(json_files, desc="Analyzing Cityscapes images", unit="files") as pbar:
        for path in pbar:
            # Collect all labels for analysis
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    objects = data.get('objects', [])
                    for obj in objects:
                        label = obj.get('label', '').strip()
                        if label:
                            unique_labels.add(label)
            except:
                pass

            # Parse annotation for target objects
            instances, img_dims = parse_json(str(path))

            # Skip if no target objects found
            if not any(instances.values()):
                continue

            files_with_targets += 1
            pbar.set_postfix({"found": files_with_targets})

            # Extract path components for organization
            # Cityscapes typically follows: city/split/filename pattern
            rel_path = path.relative_to(root)
            parts = rel_path.parts

            # Parse Cityscapes filename structure
            # Format: city_sequence_frame_*_polygons.json
            filename_base = path.stem
            if '_polygons' in filename_base:
                filename_base = filename_base.replace('_polygons', '')

            # Extract city and split information from path
            city = parts[-2] if len(parts) >= 2 else 'unknown'
            split = parts[-3] if len(parts) >= 3 else 'unknown'

            # Create result entry
            result = {
                'split': split,
                'city': city,
                'filename': filename_base,
                'image_width': img_dims[0],
                'image_height': img_dims[1]
            }

            # Add instance data with consistent naming
            for obj_type in ['person', 'building', 'car']:
                instance_list = instances[obj_type]
                for i, proportion in enumerate(instance_list, 1):
                    key = f'{obj_type}_{i}_proportion'
                    result[key] = f"{proportion:.6f}"

            results.append(result)

    print(f"\nProcessed {len(json_files)} files, found {files_with_targets} with target objects")

    # Show relevant labels found in dataset
    print("\nRelevant object labels found:")
    all_targets = CAR_NAMES | PERSON_NAMES | BUILDING_NAMES
    found_targets = unique_labels & all_targets
    other_vehicle_labels = [label for label in unique_labels
                            if any(vehicle_word in label.lower()
                                   for vehicle_word in ['car', 'truck', 'bus', 'vehicle'])]

    if found_targets:
        print(f"  Target labels: {sorted(found_targets)}")
    if other_vehicle_labels:
        print(f"  Other vehicle labels: {sorted(set(other_vehicle_labels) - found_targets)[:5]}...")

    return results


def save_csv(results: List[Dict], output_path: str):
    """Write analysis results to CSV file with logical column ordering."""
    if not results:
        print("No results to write.")
        return

    # Collect all possible columns
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Order columns logically for Cityscapes structure
    base_cols = ['split', 'city', 'filename', 'image_width', 'image_height']
    person_cols = sorted([k for k in all_keys if k.startswith('person_')])
    building_cols = sorted([k for k in all_keys if k.startswith('building_')])
    car_cols = sorted([k for k in all_keys if k.startswith('car_')])

    fieldnames = base_cols + person_cols + building_cols + car_cols

    # Write CSV with organized structure
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


def show_stats(results: List[Dict]):
    """Print comprehensive summary statistics about analysis results."""
    if not results:
        return

    # Collect instances by type for statistical analysis
    person_instances = []
    building_instances = []
    car_instances = []

    for r in results:
        for k, v in r.items():
            if k.startswith('person_') and k.endswith('_proportion'):
                person_instances.append(float(v))
            elif k.startswith('building_') and k.endswith('_proportion'):
                building_instances.append(float(v))
            elif k.startswith('car_') and k.endswith('_proportion'):
                car_instances.append(float(v))

    print("\n=== Cityscapes Analysis Summary ===")

    # Calculate and display statistics for each object type
    for name, instances in [("Person", person_instances),
                            ("Building", building_instances),
                            ("Car", car_instances)]:
        if instances:
            avg_area = sum(instances) / len(instances)
            min_area = min(instances)
            max_area = max(instances)

            print(f"\n{name} instances: {len(instances)}")
            print(f"  Average area proportion: {avg_area:.4%}")
            print(f"  Size range: {min_area:.4%} - {max_area:.4%}")

            # Additional insights for urban scene understanding
            large_instances = [x for x in instances if x > 0.01]  # > 1% of image
            if large_instances:
                print(f"  Large instances (>1% of image): {len(large_instances)}")
        else:
            print(f"\n{name} instances: 0 (not found in dataset)")

    # Dataset composition insights
    cities = set(r['city'] for r in results)
    splits = set(r['split'] for r in results)

    print(f"\nDataset composition:")
    print(f"  Cities represented: {len(cities)}")
    print(f"  Data splits: {sorted(splits)}")

    if len(cities) > 1:
        print(f"  Sample cities: {sorted(list(cities))[:5]}{'...' if len(cities) > 5 else ''}")


def main():
    """Main function to execute the Cityscapes analysis workflow."""
    print("=== Cityscapes Object Analyzer ===\n")

    dataset_root = input("Enter Cityscapes dataset root path: ").strip()
    output_csv = "cityscapes_analysis.csv"

    # Validate dataset path exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path '{dataset_root}' does not exist.")
        return

    print("\nAnalyzing Cityscapes dataset for: person, building, car objects")
    print("Looking for *_polygons.json annotation files...")
    print("This analysis may take several minutes for large datasets.\n")

    # Execute the analysis pipeline
    results = analyze_dataset(dataset_root)

    if not results:
        print("\nNo target objects found. Please verify:")
        print("- Dataset path contains Cityscapes annotation files")
        print("- Files follow *_polygons.json naming convention")
        print("- Annotations contain person, building, or car labels")
        return

    print(f"\nAnalysis complete! Found {len(results)} images with target objects.")

    # Generate insights and save results
    show_stats(results)
    save_csv(results, output_csv)

    print(f"\nResults saved to {output_csv}")
    print("Analysis includes area proportions for each object instance.")


if __name__ == "__main__":
    main()