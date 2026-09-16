import os
import zipfile
import datetime

JOURNAL = "journals/myJournal.txt"
MAX_SIZE = 50000

def archive_if_needed(memory):
    if not os.path.exists(JOURNAL):
        return

    text = open(JOURNAL, "r", encoding="utf-8").read()

    if len(text) < MAX_SIZE:
        return

    os.makedirs("journals/archive", exist_ok=True)

    filename = f"journals/archive/archive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    with zipfile.ZipFile(filename, "w") as zipf:
        zipf.write(JOURNAL, "myJournal.txt")

    with open(JOURNAL, "w", encoding="utf-8") as f:
        f.write("===== COMPRESSED MEMORY =====\n\n" + memory)