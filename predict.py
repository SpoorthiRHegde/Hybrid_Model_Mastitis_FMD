import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("Select the condition to check:")
    print("1. Mastitis")
    print("2. FMD (Foot and Mouth Disease)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        script = "Mastitis_Detection\mastitis.py"
    elif choice == "2":
        script = "FMD_Detection/fmd.py"
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")
        return

    # Build the path to the selected script
    script_path = os.path.join(os.path.dirname(__file__), script)

    # Run the selected script
    subprocess.run([sys.executable, "-W", "ignore::UserWarning", script_path])

if __name__ == "__main__":
    main()
