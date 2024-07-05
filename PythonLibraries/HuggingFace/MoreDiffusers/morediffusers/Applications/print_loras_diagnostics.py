def print_loras_diagnostics(duration, pipe):

    print("-------------------------------------------------------------------")
    print(f"Completed loading LoRAs, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    print("\n LoRAs: \n")
    print(pipe.get_active_adapters())
    print(pipe.get_list_adapters())
