#!/usr/bin/env python3
"""
ADE20K Object Analyzer
Analyzes JSON annotation files to find instances of cars, people, and buildings.
Calculates the proportion of each instance relative to total image area.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Target object names based on ADE20K's objectInfo150 naming convention
CAR_NAMES = {'car, auto, automobile, machine, motorcar', 'truck', 'bus', 'van'}
PERSON_NAMES = {'person, individual, someone, somebody, mortal, soul'}
BUILDING_NAMES = {'building, edifice', 'house', 'skyscraper'}


def load_objects(csv_path: str) -> Dict[str, int]:
    """Load objectInfo150.csv to get valid object names."""
    object_map = {}

    if not os.path.exists(csv_path):
        return object_map

    # Try different encodings for compatibility
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('Name', '').strip()
                    idx = row.get('Idx', 0)
                    if name:
                        object_map[name] = int(idx)
            print(f"Loaded {len(object_map)} object categories (encoding: {encoding})")
            return object_map
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]:
                print(f"Warning: Could not load objectInfo150.csv: {e}")

    return object_map


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


def parse_json(json_path: str, valid_objects: Optional[Dict[str, int]] = None) -> Tuple[
    Dict[str, List[float]], Tuple[int, int]]:
    """Parse ADE20K JSON file to extract object instances and areas."""
    # Try multiple encodings for robustness
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'utf-16']
    data = None

    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            break
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            print(f"JSON error in {json_path}: {e}")
            return {}, (0, 0)
        except Exception as e:
            if encoding == encodings[-1]:
                print(f"Error reading {json_path}: {e}")
                return {}, (0, 0)

    # Fallback with error handling
    if data is None:
        try:
            with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")
            return {}, (0, 0)

    # Initialize containers
    cars = []
    people = []
    buildings = []

    # Get annotation data
    annotation = data.get('annotation', {})

    # Get image dimensions [width, height, channels]
    imsize = annotation.get('imsize', [])
    if len(imsize) >= 2:
        img_width = imsize[0]
        img_height = imsize[1]
    else:
        # Fallback format
        img_info = data.get('imgsize', {})
        img_width = img_info.get('ncols', 0)
        img_height = img_info.get('nrows', 0)

    total_area = img_width * img_height
    if total_area == 0:
        return {}, (0, 0)

    # Parse objects
    objects = annotation.get('object', [])

    for obj in objects:
        obj_name = obj.get('name', '').strip()

        # Skip if not in valid objects list
        if valid_objects and obj_name not in valid_objects:
            continue

        # Get polygon coordinates
        polygon_data = obj.get('polygon', {})
        x_coords = polygon_data.get('x', [])
        y_coords = polygon_data.get('y', [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        # Combine coordinates
        polygon = [[x_coords[i], y_coords[i]] for i in range(len(x_coords))]

        # Calculate area proportion
        area = calc_area(polygon)
        proportion = area / total_area

        # Classify object
        if obj_name in CAR_NAMES:
            cars.append(proportion)
        elif obj_name in PERSON_NAMES:
            people.append(proportion)
        elif obj_name in BUILDING_NAMES:
            buildings.append(proportion)

    return {
               'car': cars,
               'person': people,
               'building': buildings
           }, (img_width, img_height)


def analyze_dataset(root_path: str, object_info_path: Optional[str] = None) -> List[Dict]:
    """Traverse ADE20K dataset and collect statistics for all JSON files."""
    results = []
    root = Path(root_path)

    # Load valid object names
    valid_objects = None
    if object_info_path:
        object_info_file = os.path.join(object_info_path, 'objectInfo150.csv')
        valid_objects = load_objects(object_info_file)

    # Find all JSON files first to show accurate progress
    json_files = [p for p in root.rglob('*.json')
                  if 'index' not in p.name and 'meta' not in p.name]

    files_with_targets = 0
    unique_objects = set()

    # Process files with progress bar
    with tqdm(json_files, desc="Analyzing images", unit="files") as pbar:
        for path in pbar:
            # Collect object names for analysis
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                    objects = data.get('objects', [])
                    for obj in objects:
                        obj_name = obj.get('name', '').strip()
                        if obj_name:
                            unique_objects.add(obj_name)
            except:
                pass

            # Parse annotation for target objects
            instances, img_dims = parse_json(str(path), valid_objects)

            # Skip if no target objects found
            if not any(instances.values()):
                continue

            files_with_targets += 1
            pbar.set_postfix({"found": files_with_targets})

            # Get path hierarchy
            rel_path = path.relative_to(root)
            parts = rel_path.parts

            level1 = parts[0] if len(parts) > 0 else 'unknown'
            level2 = parts[1] if len(parts) > 1 else 'unknown'

            # Handle deeper structures
            if len(parts) > 3:
                level2 = '/'.join(parts[1:-1])

            # Create result entry
            result = {
                'level1': level1,
                'level2': level2,
                'filename': path.stem,
                'image_width': img_dims[0],
                'image_height': img_dims[1]
            }

            # Add instance data
            for obj_type in ['person', 'building', 'car']:
                instance_list = instances[obj_type]
                for i, proportion in enumerate(instance_list, 1):
                    key = f'{obj_type}_{i}_proportion'
                    result[key] = f"{proportion:.6f}"

            results.append(result)

    print(f"\nProcessed {len(json_files)} files, found {files_with_targets} with target objects")

    # Show sample of relevant object names
    print("\nRelevant object names found:")
    keywords = ['car', 'person', 'building', 'truck', 'bus', 'van', 'house']
    for keyword in keywords:
        matches = [name for name in unique_objects if keyword in name.lower()]
        if matches:
            print(f"  {keyword}: {matches[:3]}{'...' if len(matches) > 3 else ''}")

    return results


def save_csv(results: List[Dict], output_path: str):
    """Write analysis results to CSV file."""
    if not results:
        print("No results to write.")
        return

    # Get all columns
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Order columns logically
    base_cols = ['level1', 'level2', 'filename', 'image_width', 'image_height']
    person_cols = sorted([k for k in all_keys if k.startswith('person_')])
    building_cols = sorted([k for k in all_keys if k.startswith('building_')])
    car_cols = sorted([k for k in all_keys if k.startswith('car_')])

    fieldnames = base_cols + person_cols + building_cols + car_cols

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


def show_stats(results: List[Dict]):
    """Print summary statistics about the analysis results."""
    if not results:
        return

    # Collect all instances
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

    print("\n=== Summary Statistics ===")

    # Show stats for each object type
    for name, instances in [("Person", person_instances),
                            ("Building", building_instances),
                            ("Car", car_instances)]:
        if instances:
            avg = sum(instances) / len(instances)
            print(f"\n{name} instances: {len(instances)}")
            print(f"  Average area: {avg:.4%}")
            print(f"  Range: {min(instances):.4%} - {max(instances):.4%}")


def main():
    """Main function to run the analysis."""
    print("=== ADE20K Object Analyzer ===\n")

    dataset_root = input("Enter ADE20K dataset root path: ").strip()
    output_csv = "ade20k_statistics.csv"

    # Validate path
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path '{dataset_root}' does not exist.")
        return

    print("\nAnalyzing ADE20K dataset for: person, building, car objects")
    print("This may take several minutes...\n")

    # Run analysis
    results = analyze_dataset(dataset_root, dataset_root)

    print(f"\nAnalysis complete! Found {len(results)} images with target objects.")

    # Show statistics and save results
    show_stats(results)
    save_csv(results, output_csv)


if __name__ == "__main__":
    main()
