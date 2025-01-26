import csv

# Input and output file paths
input_file = "origin.csv"
output_file = "basic_data.csv"

# Number of values to group in each row
group_size = 300

# Process the file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # Read all lines into memory
    data = [line.strip() for line in infile]

    # Process each line
    for i in range(2300000):
        first_value = data[i]
        group = data[i+1:i+1+group_size]
        group_str = ",".join(group)
        # Write the result line to the file
        outfile.write(f"{first_value},{group_str}\n")

print(f"Processed data has been written to {output_file}.")