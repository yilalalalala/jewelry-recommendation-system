import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

FILE_TO_WATCH = "ready_for_ml_adjusted_if.csv"
SCRIPT_TO_RUN = "export_ml_to_supabase.py"  
DEBOUNCE_SECONDS = 5 

class RetrainHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_triggered = 0

    def on_modified(self, event):
        # Only trigger if the specific CSV was changed
        if event.src_path.endswith(FILE_TO_WATCH):
            current_time = time.time()
            
            # Prevent multiple triggers if the file is saved in chunks
            if current_time - self.last_triggered > DEBOUNCE_SECONDS:
                print(f"\n[DETECTED] {FILE_TO_WATCH} has changed.")
                print(f"Waiting {DEBOUNCE_SECONDS}s for file to finalize...")
                time.sleep(DEBOUNCE_SECONDS)
                
                self.run_export_script()
                self.last_triggered = time.time()

    def run_export_script(self):
        print(f"--- Launching Export: {SCRIPT_TO_RUN} ---")
        try:
            # Runs your existing export code as a separate process
            subprocess.run(["python", SCRIPT_TO_RUN], check=True)
            print("--- Export Finished Successfully ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Export Failed with error: {e} ---")
        print("\nListening for next change...")

if __name__ == "__main__":
    event_handler = RetrainHandler()
    observer = Observer()
    # Watch the current directory (.)
    observer.schedule(event_handler, path='.', recursive=False)
    
    print(f"Watcher started! Waiting for changes to {FILE_TO_WATCH}...")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Watcher stopped.")
    observer.join()
