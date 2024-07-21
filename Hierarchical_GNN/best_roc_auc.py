import re
import sys

def find_largest_best_test_value(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        best_test_pattern = re.compile(r"Best ROC Test: (\d+(\.\d+)?)")
        best_tests = []
        for i, line in enumerate(lines):
            match = best_test_pattern.search(line)
            if match:
                value = float(match.group(1))
                best_tests.append((value, i))
        if not best_tests:
            print("No 'Best ROC AUC: X%' lines found.")
            return
        # Find the maximum value and its index
        max_value, max_index = max(best_tests, key=lambda x: x[0])
        # Print the results
        print(f"Largest value: {max_value}%")
        print(f"Line number: {max_index + 1}")  # +1 to convert from 0-based to 1-based indexing
        print("Two lines above the largest value line:")
        print(".")
        print(f"All {len(best_tests)} test values: ")
        print(best_tests)
        if max_index >= 2:
            print(lines[max_index - 2].strip())
        if max_index >= 1:
            print(lines[max_index - 1].strip())
        print(lines[max_index].strip())  # This is the line with the largest value
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    find_largest_best_test_value(file_path)
