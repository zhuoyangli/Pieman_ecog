# batch run analysis and plot code
import subprocess

subjects = [
    "sub-004",
    "sub-005",
    "sub-007",
    "sub-015",
    "sub-016",
    "sub-018",
    "sub-019",
    "sub-020",
    "sub-021",
]

for sub in subjects:
    print(f"Running analysis for {sub}...")
    # subprocess.run(['python', 'train_EM_lag.py', sub])
    subprocess.run(
        [
            "python",
            "plot_EM.py",
            "--subject",
            sub,
            "--session",
            "ses-001",
            "--gpt",
            "gpt2",
        ]
    )
