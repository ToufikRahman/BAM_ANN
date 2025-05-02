# Function to save experimental results
def save_results(results, filename="experiment_results.txt"):
    with open(filename, "a") as file:  # 'a' mode opens the file for appending
        file.write(str(results) + "\n")
        # file.write("-" * 50 + "\n")  # Separator line for readability
        print('Saved results')


# Function to read experimental results
def read_results(filename="experiment_results.txt"):
    with open(filename, "r") as file:
        results = file.readlines()
    
    # Display results in a structured format
    for line in results:
        print(line.strip())

