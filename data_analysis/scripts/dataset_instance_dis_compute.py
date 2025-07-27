#!/usr/bin/env python3
"""
Multi-Dataset Instance Proximity Analyzer
Analyzes minimum distances between same-class instances across multiple CV datasets.
Supports COCO, Cityscapes, ADE20K, and CBL (LabelMe) formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from itertools import combinations
from tqdm import tqdm


class ProximityAnalyzer:
    """
    Analyzes minimum distances between same-class object instances.
    Supports multiple dataset formats with unified interface.
    """

    def __init__(self):
        # Distance thresholds for analysis (in pixels)
        self.thresholds = [10, 20, 30, 40]
        self.target_classes = ['person', 'car', 'building']

        # COCO category mappings
        self.coco_categories = {
            1: 'person',
            3: 'car'
            # Note: COCO doesn't have building category
        }

    def calc_min_distance(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate minimum distance between two polygons."""
        if poly1.is_empty or poly2.is_empty:
            return float('inf')

        p1, p2 = nearest_points(poly1, poly2)
        return p1.distance(p2)

    def find_nearest_neighbor_distances(self, polygons: List[Polygon]) -> List[float]:
        """
        For each polygon, find distance to its nearest same-class neighbor.
        Returns list of minimum distances for each instance.
        """
        if len(polygons) < 2:
            return []

        min_distances = []

        for i, target_poly in enumerate(polygons):
            distances_to_others = []

            for j, other_poly in enumerate(polygons):
                if i != j:  # Skip self-comparison
                    dist = self.calc_min_distance(target_poly, other_poly)
                    if dist != float('inf'):
                        distances_to_others.append(dist)

            if distances_to_others:
                min_distances.append(min(distances_to_others))

        return min_distances

    def analyze_ade20k(self, root_path: str, splits: List[str],
                       target_class: str) -> List[float]:
        """Analyze ADE20K dataset annotations."""
        print(f"Analyzing ADE20K dataset: {target_class}")

        # ADE20K specific class mappings
        ade20k_class_map = {
            'person': {'person, individual, someone, somebody, mortal, soul'},
            'car': {'car, auto, automobile, machine, motorcar', 'truck', 'bus', 'van'},
            'building': {'building, edifice', 'house', 'skyscraper'}
        }

        target_names = ade20k_class_map.get(target_class, {target_class})
        all_distances = []

        root = Path(root_path)
        json_files = list(root.rglob('*.json'))

        with tqdm(json_files, desc=f"Processing ADE20K {target_class}") as pbar:
            for json_file in pbar:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract polygons for target class
                    polygons = []
                    annotation = data.get('annotation', {})
                    objects = annotation.get('object', [])

                    for obj in objects:
                        obj_name = obj.get('name', '').strip()
                        if obj_name in target_names:
                            polygon_data = obj.get('polygon', {})
                            x_coords = polygon_data.get('x', [])
                            y_coords = polygon_data.get('y', [])

                            if len(x_coords) >= 3 and len(x_coords) == len(y_coords):
                                points = list(zip(x_coords, y_coords))
                                polygons.append(Polygon(points))

                    # Calculate nearest neighbor distances
                    distances = self.find_nearest_neighbor_distances(polygons)
                    all_distances.extend(distances)

                except Exception as e:
                    continue  # Skip problematic files

        return all_distances

    def analyze_cityscapes(self, root_path: str, splits: List[str],
                           target_class: str) -> List[float]:
        """Analyze Cityscapes dataset annotations."""
        print(f"Analyzing Cityscapes dataset: {target_class}")

        all_distances = []
        root = Path(root_path)

        for split in splits:
            split_path = root / split
            if not split_path.exists():
                continue

            json_files = list(split_path.rglob('*_polygons.json'))

            with tqdm(json_files, desc=f"Processing {split}") as pbar:
                for json_file in pbar:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Extract polygons for target class
                        polygons = []
                        for obj in data.get('objects', []):
                            if obj.get('label') == target_class:
                                polygon_coords = obj.get('polygon', [])
                                if len(polygon_coords) >= 3:
                                    polygons.append(Polygon(polygon_coords))

                        # Calculate nearest neighbor distances
                        distances = self.find_nearest_neighbor_distances(polygons)
                        all_distances.extend(distances)

                    except Exception as e:
                        continue

        return all_distances

    def analyze_coco(self, annotation_path: str, target_class: str) -> List[float]:
        """Analyze COCO dataset annotations."""
        print(f"Analyzing COCO dataset: {target_class}")

        if not Path(annotation_path).exists():
            print(f"COCO annotation file not found: {annotation_path}")
            return []

        # Find target category ID
        target_category_ids = set()
        for cat_id, cat_name in self.coco_categories.items():
            if cat_name == target_class:
                target_category_ids.add(cat_id)

        if not target_category_ids:
            print(f"Target class '{target_class}' not found in COCO categories")
            return []

        with open(annotation_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            if ann['category_id'] in target_category_ids:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)

        all_distances = []

        with tqdm(image_annotations.items(), desc="Processing images") as pbar:
            for image_id, annotations in pbar:
                polygons = []

                for ann in annotations:
                    segmentation = ann.get('segmentation', [])
                    if isinstance(segmentation, list):
                        for seg in segmentation:
                            if isinstance(seg, list) and len(seg) >= 6:
                                # Convert flat coordinates to point pairs
                                points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                                if len(points) >= 3:
                                    polygons.append(Polygon(points))

                # Calculate nearest neighbor distances
                distances = self.find_nearest_neighbor_distances(polygons)
                all_distances.extend(distances)

        return all_distances

    def analyze_cbl(self, root_path: str, splits: List[str]) -> List[float]:
        """Analyze CBL dataset (LabelMe format) annotations."""
        print("Analyzing CBL dataset")

        all_distances = []
        root = Path(root_path)

        for split in splits:
            split_path = root / split
            if not split_path.exists():
                continue

            json_files = list(split_path.glob('*.json'))

            with tqdm(json_files, desc=f"Processing {split}") as pbar:
                for json_file in pbar:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Extract all polygons (CBL focuses on buildings)
                        polygons = []
                        for shape in data.get('shapes', []):
                            points = shape.get('points', [])
                            if len(points) >= 3:
                                polygons.append(Polygon(points))

                        # Calculate nearest neighbor distances
                        distances = self.find_nearest_neighbor_distances(polygons)
                        all_distances.extend(distances)

                    except Exception as e:
                        continue

        return all_distances

    def calculate_distribution_stats(self, distances: List[float]) -> Dict:
        """Calculate distance distribution statistics."""
        if not distances:
            return {f'count_{t1}-{t2}': 0 for t1, t2 in
                    zip([0] + self.thresholds[:-1], self.thresholds)} | \
                   {f'ratio_{t1}-{t2}': 0.0 for t1, t2 in
                    zip([0] + self.thresholds[:-1], self.thresholds)}

        distances = np.array(distances)
        total_count = len(distances)

        stats = {}
        threshold_pairs = list(zip([0] + self.thresholds[:-1], self.thresholds))

        for t1, t2 in threshold_pairs:
            count = np.sum((distances >= t1) & (distances < t2))
            ratio = count / total_count if total_count > 0 else 0.0

            stats[f'count_{t1}-{t2}'] = count
            stats[f'ratio_{t1}-{t2}'] = ratio

        return stats

    def save_results(self, results: Dict[str, Dict], output_path: str):
        """Save analysis results to CSV file."""
        if not results:
            print("No results to save")
            return

        # Prepare CSV data
        rows = []
        for dataset_class, stats in results.items():
            row = {'dataset_class': dataset_class}
            row.update(stats)
            rows.append(row)

        if not rows:
            return

        # Write CSV
        fieldnames = ['dataset_class'] + list(rows[0].keys())[1:]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results saved to: {output_path}")

    def run_analysis(self, dataset_configs: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Run proximity analysis on specified datasets.

        Args:
            dataset_configs: Dictionary with dataset configurations
                Format: {
                    'dataset_name_class': {
                        'type': 'coco|cityscapes|ade20k|cbl',
                        'path': 'dataset_path',
                        'splits': ['train', 'val'],  # Optional for COCO
                        'target_class': 'person|car|building'  # Optional for CBL
                    }
                }
        """
        results = {}

        for config_name, config in dataset_configs.items():
            dataset_type = config['type']
            dataset_path = config['path']

            print(f"\n--- Processing {config_name} ---")

            try:
                if dataset_type == 'ade20k':
                    distances = self.analyze_ade20k(
                        dataset_path,
                        config.get('splits', ['train', 'val']),
                        config['target_class']
                    )
                elif dataset_type == 'cityscapes':
                    distances = self.analyze_cityscapes(
                        dataset_path,
                        config.get('splits', ['train', 'val']),
                        config['target_class']
                    )
                elif dataset_type == 'coco':
                    distances = self.analyze_coco(
                        dataset_path,
                        config['target_class']
                    )
                elif dataset_type == 'cbl':
                    distances = self.analyze_cbl(
                        dataset_path,
                        config.get('splits', ['train', 'val'])
                    )
                else:
                    print(f"Unknown dataset type: {dataset_type}")
                    continue

                # Calculate statistics
                stats = self.calculate_distribution_stats(distances)
                stats['total_instances'] = len(distances)
                results[config_name] = stats

                print(f"Processed {len(distances)} instances")

            except Exception as e:
                print(f"Error processing {config_name}: {e}")
                continue

        return results


def display_menu():
    """Display available dataset options to user."""
    print("\n=== Multi-Dataset Instance Proximity Analysis ===")
    print("Available datasets:")
    print("1. COCO (analyzes: person + car)")
    print("2. Cityscapes (analyzes: person + car)")
    print("3. ADE20K (analyzes: person + car)")
    print("4. CBL (analyzes: building)")
    print("5. Exit")


def get_user_choice():
    """Get user's dataset selection."""
    while True:
        try:
            choice = int(input("\nSelect dataset (1-5): "))
            if 1 <= choice <= 5:
                return choice
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")


def get_dataset_path(dataset_name: str) -> str:
    """Get dataset path from user input."""
    while True:
        path = input(f"\nEnter path for {dataset_name} dataset: ").strip()
        if Path(path).exists():
            return path
        else:
            print(f"Path does not exist: {path}")
            print("Please enter a valid path.")


def get_splits() -> List[str]:
    """Get data splits to analyze."""
    print("\nWhich data splits would you like to analyze?")
    print("Available: train, val, test")
    splits_input = input("Enter splits (comma-separated, e.g., 'train,val'): ").strip()

    # Parse and validate splits
    available_splits = ['train', 'val', 'test']
    splits = [s.strip().lower() for s in splits_input.split(',')]
    valid_splits = [s for s in splits if s in available_splits]

    if not valid_splits:
        print("No valid splits provided. Using default: train, val")
        return ['train', 'val']

    return valid_splits


def configure_dataset_analysis():
    """Interactive configuration of dataset analysis."""
    while True:
        display_menu()
        choice = get_user_choice()

        if choice == 5:  # Exit
            print("Exiting analysis.")
            return None

        # Map choice to dataset configuration
        configs = {}  # Will hold multiple configurations for person+car datasets

        if choice == 1:  # COCO
            dataset_path = get_dataset_path("COCO annotation file")

            # Create configurations for both person and car
            configs['coco_person'] = {
                'type': 'coco',
                'path': dataset_path,
                'target_class': 'person'
            }
            configs['coco_car'] = {
                'type': 'coco',
                'path': dataset_path,
                'target_class': 'car'
            }

        elif choice == 2:  # Cityscapes
            dataset_path = get_dataset_path("Cityscapes gtFine")
            splits = get_splits()

            # Create configurations for both person and car
            configs['cityscapes_person'] = {
                'type': 'cityscapes',
                'path': dataset_path,
                'splits': splits,
                'target_class': 'person'
            }
            configs['cityscapes_car'] = {
                'type': 'cityscapes',
                'path': dataset_path,
                'splits': splits,
                'target_class': 'car'
            }

        elif choice == 3:  # ADE20K
            dataset_path = get_dataset_path("ADE20K")
            splits = get_splits()

            # Create configurations for both person and car
            configs['ade20k_person'] = {
                'type': 'ade20k',
                'path': dataset_path,
                'splits': splits,
                'target_class': 'person'
            }
            configs['ade20k_car'] = {
                'type': 'ade20k',
                'path': dataset_path,
                'splits': splits,
                'target_class': 'car'
            }

        elif choice == 4:  # CBL
            dataset_path = get_dataset_path("CBL")
            splits = get_splits()

            # CBL only analyzes buildings
            configs['cbl_building'] = {
                'type': 'cbl',
                'path': dataset_path,
                'splits': splits
            }

        # Display configuration summary
        print(f"\nConfiguration summary:")
        for config_name, config in configs.items():
            print(f"\nAnalysis: {config_name}")
            print(f"  Path: {config['path']}")
            if 'target_class' in config:
                print(f"  Target class: {config['target_class']}")
            if 'splits' in config:
                print(f"  Splits: {', '.join(config['splits'])}")

        confirm = input(f"\nProceed with analysis of {len(configs)} configuration(s)? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return configs
        else:
            print("Configuration cancelled. Please try again.")


def main():
    """Main execution function with interactive dataset selection."""
    analyzer = ProximityAnalyzer()

    # Get user configuration
    dataset_configs = configure_dataset_analysis()

    if dataset_configs is None:
        return

    print(f"\n=== Starting Analysis of {len(dataset_configs)} Configuration(s) ===")

    # Run analysis on all configurations
    results = analyzer.run_analysis(dataset_configs)

    if not results:
        print("No results generated. Analysis may have failed.")
        return

    # Generate output filename based on dataset type
    first_config = list(dataset_configs.keys())[0]
    dataset_type = first_config.split('_')[0]  # Extract dataset name
    output_filename = f'proximity_analysis_{dataset_type}_complete.csv'

    # Save results
    analyzer.save_results(results, output_filename)

    # Print comprehensive summary
    print("\n=== Analysis Summary ===")
    for dataset_class, stats in results.items():
        print(f"\n{dataset_class}:")
        print(f"  Total instances analyzed: {stats.get('total_instances', 0)}")

        if stats.get('total_instances', 0) > 0:
            print(f"  Distance distribution:")
            for i in range(len(analyzer.thresholds)):
                if i == 0:
                    range_key = f'ratio_0-{analyzer.thresholds[i]}'
                    count_key = f'count_0-{analyzer.thresholds[i]}'
                    range_desc = f"0-{analyzer.thresholds[i]}px"
                else:
                    range_key = f'ratio_{analyzer.thresholds[i - 1]}-{analyzer.thresholds[i]}'
                    count_key = f'count_{analyzer.thresholds[i - 1]}-{analyzer.thresholds[i]}'
                    range_desc = f"{analyzer.thresholds[i - 1]}-{analyzer.thresholds[i]}px"

                ratio = stats.get(range_key, 0.0)
                count = stats.get(count_key, 0)
                print(f"    {range_desc}: {count} instances ({ratio:.2%})")
        else:
            print("  No instances found for analysis.")

    print(f"\nDetailed results saved to: {output_filename}")

    # Provide comparison insights if multiple classes were analyzed
    if len(results) > 1:
        print(f"\n=== Cross-Class Comparison ===")
        print("You can now compare proximity patterns between:")

        for config_name in results.keys():
            dataset, class_name = config_name.split('_', 1)
            total_instances = results[config_name].get('total_instances', 0)
            print(f"  - {class_name.title()}: {total_instances} instances")

        print("\nThis enables analysis of whether people or cars show different clustering behaviors.")

    # Ask if user wants to analyze another dataset
    another = input("\nAnalyze another dataset? (y/n): ").strip().lower()
    if another in ['y', 'yes']:
        main()  # Recursive call for another analysis


if __name__ == "__main__":
    main()