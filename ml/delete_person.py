import pickle
import os

DATABASE_FILE = "data/face_database.pkl"

if not os.path.exists(DATABASE_FILE):
    print("❌ Database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)

print("\nRegistered people:")

names = list(database.keys())

for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

choice = int(input("\nEnter the number to delete: "))

if choice < 1 or choice > len(names):
    print("❌ Invalid choice.")
    exit()

name_to_delete = names[choice - 1]

del database[name_to_delete]

with open(DATABASE_FILE, "wb") as file:
    pickle.dump(database, file)

print(f"\n✅ Deleted: {name_to_delete}")

print("\nRemaining registered people:")

for name in database:
    print(f"  - {name}")