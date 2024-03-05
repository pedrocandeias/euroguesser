import random

def load_txt_entries(txt_file_path):
    """
    Load entries from a TXT file with the format: 3,4,9,12,20,5,6.
    """
    formatted_entries = []
    with open(txt_file_path, 'r') as txt_file:
        for line in txt_file:
            parts = line.strip().split(',')
            if len(parts) == 7:  # Ensure correct number of parts
                formatted_entry = '{},{},{},{},{} - {},{}'.format(*parts[:5], *parts[5:])
                formatted_entries.append(formatted_entry)
            else:
                print(f"Invalid line format, skipping: {line.strip()}")
    return formatted_entries

def generate_random_string():
    """
    Generate a random string in the EuroMillions format: "XX,XX,XX,XX,XX - XX,XX".
    """
    main_numbers = sorted(random.sample(range(1, 51), 5))
    lucky_stars = sorted(random.sample(range(1, 13), 2))
    return '{},{},{},{},{} - {},{}'.format(*main_numbers, *lucky_stars)

def match_txt_entries(txt_entries, output_file_path, log_file_path, max_attempts=None):
    """
    Attempt to sequentially match generated strings with the TXT entries, write the final string to a separate TXT file,
    and log each match found to another TXT file, with the option for infinite attempts or a set maximum.
    """
    matched_entries = []
    attempts = 0
    feedback_interval = 1000  # How often to provide feedback
    max_attempts = None

    while len(matched_entries) < len(txt_entries) and (max_attempts is None or attempts < max_attempts):
        attempts += 1
        generated_string = generate_random_string()
        if generated_string == txt_entries[len(matched_entries)]:
            matched_entries.append(generated_string)
            match_info = f"Match found: {generated_string} (Attempt: {attempts}, Matches: {len(matched_entries)})"
            print(match_info)
            # Log each match to the specified log file
            with open(log_file_path, 'a') as log_file:
                log_file.write(match_info + '\n')
                
            if len(matched_entries) == len(txt_entries):
                final_string = generate_random_string()
                with open(output_file_path, 'a') as f:
                    f.write(final_string + '\n')
                print(f"All entries matched after {attempts} attempts. Final string written to {output_file_path}.")
                return final_string
        else:
            # Provide feedback before resetting if it's time or there were any matches
            if attempts % feedback_interval == 0 or len(matched_entries) > 0:
                print(f"Attempt {attempts}: Matched {len(matched_entries)} before resetting.")
            # Reset if unmatched
            matched_entries = []

    if max_attempts is not None and attempts == max_attempts:
        print("Max attempts reached without matching all entries.")
        return None


txt_file_path = 'euro_millions_entries.txt'
output_file_path = 'path_to_your_output_file.txt'
log_file_path = 'path_to_your_log_file.txt'  # Specify the path for the log file here

# Load the TXT entries
txt_entries = load_txt_entries(txt_file_path)


# Start the matching process with enhanced feedback and flexible max attempts
# For infinite attempts, use None or for a specific number, replace None with that number
match_txt_entries(txt_entries, output_file_path, log_file_path, max_attempts=None)