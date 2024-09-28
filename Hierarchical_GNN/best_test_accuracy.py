import re
import sys

def find_largest_best_test_value(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
        
        best_test_pattern = re.compile(r"Best test: (\d+(\.\d+)?)%")
        best_tests = []
        
        for i, line in enumerate(lines):
            match = best_test_pattern.search(line)
            if match:
                value = float(match.group(1))
                best_tests.append((value, i))
        
        if not best_tests:
            print("No 'Best test: X%' lines found.")
            return
        
        # Find the maximum value and its index
        max_value, max_index = max(best_tests, key=lambda x: x[0])
        all_results = [score for score,idx in best_tests]
        # Prepare the results
        results = [
            f"Largest value: {max_value}%",
            f"Line number: {max_index + 1}",  # +1 to convert from 0-based to 1-based indexing
            "Two lines above the largest value line:"
        ]
        if max_index >= 2:
            results.append(lines[max_index - 2].strip())
        if max_index >= 1:
            results.append(lines[max_index - 1].strip())
        results.append(lines[max_index].strip())  # This is the line with the largest value

        # Print the results
        for result in results:
            print(result)
        
        # Write the results to the output file
        with open(output_file_path, 'w') as output_file:
        	output_file.write("All Results:"+'\n')
        	# Write all values as a comma-separated line
        	all_values = ','.join(str(value) for value, _ in best_tests)
        	output_file.write(all_values + '\n')
        	for result in results:
                	output_file.write(result + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    find_largest_best_test_value(input_file_path, output_file_path)
