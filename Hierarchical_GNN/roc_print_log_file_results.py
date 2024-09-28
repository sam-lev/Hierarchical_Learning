
import os
import sys
import re

def print_last_lines_of_log_files(directory_path, output_file_path):
    try:
        with open(output_file_path, 'w') as output_file:
            # List all files in the given directory
            files = os.listdir(directory_path)
            
            # Filter files that end with ".log"
            log_files = [f for f in files if f.endswith('.log')]
            
            for log_file in log_files:
                log_file_path = os.path.join(directory_path, log_file)
                
                # Check if the path is a file
                if os.path.isfile(log_file_path):
                    with open(log_file_path, 'r') as file:
                        lines = file.readlines()
                        
                        # Print the file name
                        output_file.write(f'File: {log_file}\n')
                        print(f'File: {log_file}')
                        
                        # Print the last 4 lines
                        last_lines = lines[-4:]
                        for line in last_lines:
                            output_file.write(line)
                            print(line, end='')
                        
                        # Print a separator
                        output_file.write('\n' + '-'*40 + '\n\n')
                        print('\n' + '-'*40 + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")

def find_largest_best_test_value(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
            print(1) ##
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


def find_largest_best_roc_value(file_path, output_file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        best_test_pattern = re.compile(r"Test RoC: (\d+(\.\d+)?)") #"Best ROC Test: (\d+(\.\d+)?)")
        best_tests = []
        for i, line in enumerate(lines):
            match = best_test_pattern.search(line)
            if match:
                value = float(match.group(1))
                best_tests.append((value, i))
        if not best_tests:
            print("No 'Best ROC AUC: X%' lines found.")
            return
            
    #     # Find the maximum value and its index
    #     max_value, max_index = max(best_tests, key=lambda x: x[0])
    #     # Print the results
    #     print(f"Largest value: {max_value}%")
    #     print(f"Line number: {max_index + 1}")  # +1 to convert from 0-based to 1-based indexing
    #     print("Two lines above the largest value line:")
    #     print(".")
    #     print(f"All {len(best_tests)} test values: ")
    #     print(best_tests)
    #     if max_index >= 2:
    #         print(lines[max_index - 2].strip())
    #     if max_index >= 1:
    #         print(lines[max_index - 1].strip())
    #     print(lines[max_index].strip())  # This is the line with the largest value
    # except Exception as e:
    #     print(f"An error occurred: {e}")
        
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

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script_name.py <file_path>")
#         sys.exit(1)
#     file_path = sys.argv[1]
#     find_largest_best_test_value(file_path)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <directory_path> <output_file_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    output_file_path = sys.argv[2]

    print_last_lines_of_log_files(directory_path, output_file_path)

    print("Summary of all results")
    print(output_file_path)
    directory, filename = os.path.split(output_file_path)
    parent_directory = os.path.dirname(directory)
    file_name_without_ext = os.path.splitext(filename)[0]
    new_filename = file_name_without_ext + "_best_roc_auc.txt"
    print("writing best ROC AUC to:")
    print(new_filename)
    new_path = os.path.join(parent_directory, 'results', new_filename)
    print("saving optimal test roc aucto ", new_path)

    find_largest_best_roc_value(output_file_path, new_path)
