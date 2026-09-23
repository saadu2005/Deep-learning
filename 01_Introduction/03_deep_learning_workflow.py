"""File purpose: Present the main steps in a typical deep-learning project.

Explanation: The script prints a checklist from defining a problem and preparing data through evaluation, tuning, saving, and deployment.

Real-life example: For a fruit-quality checker, the workflow would gather fruit photos, label them, train a model, test it on unseen photos, then connect it to a sorting station.
"""

# ==========================================
# Deep Learning Workflow
# ==========================================

workflow = [
    "1. Define the problem",
    "2. Collect data",
    "3. Clean and preprocess data",
    "4. Split the data",
    "5. Build the neural network",
    "6. Train the model",
    "7. Evaluate the model",
    "8. Tune the model",
    "9. Save and deploy the model"
]

for step in workflow:
    print(step)
