
import os
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <directory_path> <output_file_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    output_file_path = sys.argv[2]
    print_last_lines_of_log_files(directory_path, output_file_path)
