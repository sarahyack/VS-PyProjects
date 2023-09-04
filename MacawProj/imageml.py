import pickle

with open("C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj", "rb") as f:
    loaded_images, loaded_labels = pickle.load(f)

print("Script is starting...")

print("Loaded images shape:", loaded_images.shape)
print("Loaded labels shape:", loaded_labels.shape)
