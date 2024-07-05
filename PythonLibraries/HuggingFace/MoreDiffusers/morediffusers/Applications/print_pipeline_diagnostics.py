def print_pipeline_diagnostics(
    duration,
    pipe,
    is_scheduler_changed,
    original_scheduler_name
    ):
    changed_scheduler_name = pipe.scheduler.config._class_name

    print("-------------------------------------------------------------------")
    print(f"Completed pipeline creation, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    if is_scheduler_changed:
        print(
            "\nDiagnostic: scheduler changed, originally: ",
            original_scheduler_name,
            "\nNow: ",
            changed_scheduler_name)
    else:
        print(
            "\nDiagnostic: scheduler didn't change, originally: ",
            original_scheduler_name,
            "\nStayed: ",
            changed_scheduler_name)

    print("\nDiagnostic: pipe.unet.config.time_cond_proj_dim: ")
    print(pipe.unet.config.time_cond_proj_dim)
