import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(content)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


class Watcher:
    def __init__(self, directory_to_watch):
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    @staticmethod
    def on_modified(event):
        if not event.is_directory:
            logging.info(f"File modified: {event.src_path}")
            read_file(event.src_path);

    @staticmethod
    def on_created(event):
        if not event.is_directory:
            logging.info(f"File created: {event.src_path}")

    @staticmethod
    def on_deleted(event):
        if not event.is_directory:
            logging.info(f"File deleted: {event.src_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_to_watch>")
        sys.exit(1)

    directory_to_watch = sys.argv[1]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    w = Watcher(directory_to_watch)
    w.run()
